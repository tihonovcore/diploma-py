import tensorflow as tf

from configuration import Configuration


class TreeGenerationLoss:
    def __init__(self):
        self.all_syntax_losses = []
        self.all_predicted_kinds = []

        self.full_syntax_loss = tf.constant(0.0)
        self.full_kind_loss = tf.constant(0.0)

        self.MIN_LG = 0.00001

    def eval_full_loss(self, status):
        for ls in self.all_syntax_losses:
            self.full_syntax_loss = self.full_syntax_loss + ls
        self.full_syntax_loss = self.full_syntax_loss / Configuration.vocabulary_size
        # full_syntax_loss = full_syntax_loss / tf.constant(len(all_syntax_losses), dtype='float32')

        for prob in self.all_predicted_kinds:
            if status == "SUCC":
                self.full_kind_loss = self.full_kind_loss - tf.math.log(prob + self.MIN_LG)
            elif status == "FAIL":
                self.full_kind_loss = self.full_kind_loss - tf.math.log(1.0 - prob + self.MIN_LG)
        # full_kind_loss = full_kind_loss / tf.constant(len(all_predicted_kinds), dtype='float32')

    def print_loss(self):
        print('syntax: %.4f' % self.full_syntax_loss.numpy())
        print('kinds : %.4f' % self.full_kind_loss.numpy())

    def get_full_loss(self):
        return self.full_syntax_loss + self.full_kind_loss

    def syntax_loss(self, real, actual, weights):
        result = []
        for (a, w) in zip(actual, weights):
            result.append(tf.reduce_sum(-tf.math.log(1 - tf.gather(a, w) + self.MIN_LG)))
        return tf.convert_to_tensor(result)
