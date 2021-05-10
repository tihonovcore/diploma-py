import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from configuration import Configuration
from implementation.encoder import Encoder


class SLM(keras.Model):
    def __init__(
            self,
            vocabulary_size=Configuration.vocabulary_size,
            node_embedding_dim=Configuration.node_embedding_dim,
            path_embedding_dim=Configuration.path_embedding_dim,
            batch_size=5,
            name="structural_language_model",
            print_shape=Configuration.print_shape,
            **kwargs
    ):
        super(SLM, self).__init__(name=name, **kwargs)
        self.print_shape = print_shape

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
            initial_value=W_init(shape=(2 * path_embedding_dim, node_embedding_dim), dtype="float32"),
            trainable=True
        )
        self.C_i = tf.Variable(
            initial_value=W_init(shape=(Configuration.max_child_index, path_embedding_dim, path_embedding_dim), dtype="float32"),
            trainable=True
        )

        self.relu = layers.ReLU()
        self.soft = layers.Activation('softmax')

    def call(self, inputs):
        inputs, target_indices, type_container_id, leaf_types, root_types, json_type_containers = inputs

        if self.print_shape: print('inputs.shape: %s' % inputs.shape)

        # (samples, paths_count, len_paths) --> (samples, paths_count, len_paths, node_embedding)
        e = self.node_embedding(inputs)
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

        result = self.soft(tf.matmul(h, tf.transpose(self.node_embedding.weights[0])))
        if self.print_shape: print('result.shape: %s' % result.shape)

        return result
