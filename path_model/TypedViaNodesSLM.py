import tensorflow as tf

from configuration import Configuration
from path_model.abstract_slm import TypedSLM
from path_model.Encoder import Encoder

from tensorflow import keras
from tensorflow.keras import layers


class TypedViaNodesSLM(TypedSLM):
    """Implementation of SLM with type injection via
    composition embedding of node's kind and node's
    type: node_e = f(kind_e, type_e)
    """

    def __init__(
            self,
            vocabulary_size=Configuration.vocabulary_size,
            node_embedding_dim=Configuration.node_embedding_dim,
            path_embedding_dim=Configuration.path_embedding_dim,
            batch_size=5,  # todo: wtf is batch_size???
            **kwargs
    ):
        super().__init__(name='structural_language_model__node', **kwargs)

        self.print_shape = False

        self.node_embedding = layers.Embedding(vocabulary_size, node_embedding_dim,
                                               batch_input_shape=[batch_size, None])
        self.lstm_1 = layers.LSTM(path_embedding_dim, return_sequences=True)
        self.lstm_2 = layers.LSTM(path_embedding_dim)
        self.transformer = keras.Sequential([Encoder(), Encoder(), Encoder(), Encoder()])

        W_init = tf.random_normal_initializer()
        self.W_r = tf.Variable(
            initial_value=W_init(shape=(path_embedding_dim, path_embedding_dim), dtype="float32"),
            trainable=True,
        )
        self.W_g = tf.Variable(
            initial_value=W_init(
                shape=(2 * path_embedding_dim, Configuration.node_embedding_dim + Configuration.type_embedding_dim),
                dtype="float32"
            ), trainable=True
        )
        self.C_i = tf.Variable(
            initial_value=W_init(
                shape=(Configuration.max_child_index, path_embedding_dim, path_embedding_dim),
                dtype="float32"
            ), trainable=True
        )

        self.relu = layers.ReLU()
        self.soft = layers.Activation('softmax')

    def call(self, inputs, training=None, mask=None):
        inputs, target_indices, type_container_id, leaf_types, root_types, type_container_embeddings = inputs
        if self.print_shape: print('inputs.shape: %s' % inputs.shape)

        # (samples, paths_count, len_paths) --> (samples, paths_count, len_paths, node_embedding)
        def new_with_node(sample_index, path_index):
            def _new_with_node(x):
                kind_emb = x[0]
                node_index = x[1][0].numpy()

                current_leaf_types = leaf_types[sample_index]
                current_root_types = root_types[sample_index]
                composed_types = current_leaf_types + [current_root_types]
                type_map = composed_types[path_index]

                container_id = type_container_id[sample_index]

                if str(node_index) in type_map:
                    type_id = type_map[str(node_index)].numpy()
                else:
                    type_id = -1

                if type_id != -1 and container_id != -1:
                    type_emb = type_container_embeddings[container_id][type_id]
                else:
                    type_emb = tf.zeros(Configuration.type_embedding_dim)

                return tf.concat([kind_emb, type_emb], axis=0), x[1]

            return _new_with_node

        def new_with_path(sample_index):
            def _new_with_path(x):
                path = x[0]
                path_index = x[1][0].numpy()
                node_indices = tf.expand_dims(tf.range(path.shape[0], dtype=tf.int32), 1)

                e, _ = tf.map_fn(new_with_node(sample_index, path_index), (path, node_indices))
                return e, x[1]

            return _new_with_path

        def new_with_sample(x):
            embedding_for_composed = x[0]
            sample_index = x[1][0].numpy()

            path_indices = tf.expand_dims(tf.range(embedding_for_composed.shape[0], dtype=tf.int32), 1)
            e, _ = tf.map_fn(new_with_path(sample_index), (embedding_for_composed, path_indices))
            return e, x[1]

        e = self.node_embedding(inputs)
        sample_indices = tf.expand_dims(tf.range(e.shape[0], dtype=tf.int32), 1)
        e, _ = tf.map_fn(new_with_sample, (e, sample_indices))
        if self.print_shape: print('e.shape: %s' % e.shape)

        # (samples, paths_count, len_paths, node_embedding) --> (samples * paths_count, len_paths, node_embedding)
        paths_count = e.row_lengths()
        all_paths = e.merge_dims(0, 1)
        if self.print_shape: print('paths_count.shape: %s' % paths_count.shape)
        if self.print_shape: print('all_paths.shape: %s' % all_paths.shape)

        # (samples * paths_count, len_paths, node_embedding) --> (samples * paths_count, path_embedding)
        H = self.lstm_1(all_paths)
        H = self.lstm_2(H)

        # (samples * paths_count, path_embedding) --> (samples, paths_count, path_embedding)
        H = tf.RaggedTensor.from_row_lengths(H, paths_count)
        if self.print_shape: print('H.shape: %s' % str(H.shape))

        # extract `root_paths` from `H`
        # print(H.row_lengths())
        R = H[:, -1:, :].to_tensor()
        H = H[:, :-1, :]

        samples, _, path_emb = R.shape
        R = tf.reshape(R, (samples, path_emb))

        paths_count = list(map(lambda cnt: cnt - 1, paths_count))

        if self.print_shape: print('R.shape %s' % R.shape)
        if self.print_shape: print('H.shape %s' % H.shape)

        # (samples, None, path_embedding) -> (samples, paths_count, path_embedding)
        H = H.to_tensor()
        if self.print_shape: print('H.shape %s' % H.shape)

        # create mask to hide non-existent paths
        n = max(paths_count)

        all_masks = []
        for p in paths_count:
            ll = tf.ones((p, p))
            lr = tf.zeros((p, n - p))
            rl = tf.zeros((n - p, p))
            rr = tf.zeros((n - p, n - p))

            l = tf.concat([ll, lr], axis=1)
            r = tf.concat([rl, rr], axis=1)

            mask = tf.concat([l, r], axis=0)

            all_masks.append(mask)

        mask = tf.convert_to_tensor(all_masks)
        if self.print_shape: print('mask.shape: %s' % mask.shape)

        # run transformer with mask
        for enc in self.transformer.layers:
            enc.set_mask(mask)

        Z = self.transformer(H)
        if self.print_shape: print('Z.shape: %s' % Z.shape)

        # work with root paths
        C = tf.map_fn(lambda i: self.C_i[i], target_indices, dtype='float32')
        R = tf.reshape(R, (samples, path_emb, 1))
        R = tf.matmul(C, R)
        R = tf.reshape(R, (samples, path_emb))
        R = tf.matmul(self.relu(R), self.W_r)

        if self.print_shape: print('R.shape %s' % R.shape)

        # weighted sum of leaf paths
        paths_count_dim = max(paths_count)
        R = tf.reshape(R, (samples, path_emb, 1))
        alpha = tf.matmul(Z, R)
        alpha = tf.reshape(alpha, (samples, paths_count_dim))
        alpha = self.soft(alpha)
        R = tf.reshape(R, (samples, path_emb))
        if self.print_shape: print('alpha.shape %s' % alpha.shape)

        alpha = tf.reshape(alpha, (samples, 1, paths_count_dim))
        Z = tf.matmul(alpha, Z)
        Z = tf.reshape(Z, (samples, path_emb))
        if self.print_shape: print('Z.shape: %s' % Z.shape)

        # create vector for prediction
        ZR = tf.concat([Z, R], axis=1)
        h = tf.matmul(ZR, self.W_g)
        h = self.relu(h)
        if self.print_shape: print('h.shape: %s' % h.shape)

        # split h to `kind` and `type`
        h, t = h[:, :Configuration.node_embedding_dim], h[:, Configuration.node_embedding_dim:]
        kind_result = self.soft(tf.matmul(h, tf.transpose(self.node_embedding.weights[0])))
        if self.print_shape: print('kind_result.shape: %s' % kind_result.shape)

        # get distribution on types
        type_result = []
        for current_type_container_id, predicted_type in zip(type_container_id, t):
            all_types = tf.convert_to_tensor(type_container_embeddings[current_type_container_id])
            predicted_type = tf.reshape(predicted_type, [1] + predicted_type.shape)
            distribution = tf.matmul(predicted_type, all_types, transpose_b=True)
            distribution = self.soft(distribution)
            distribution = tf.reshape(distribution, distribution.shape[1:])
            type_result.append(distribution)

        return kind_result, type_result
