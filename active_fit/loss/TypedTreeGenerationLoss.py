import tensorflow as tf

from active_fit.loss.TreeGenerationLoss import TreeGenerationLoss
from configuration import Configuration


class TypedTreeGenerationLoss(TreeGenerationLoss):
    def __init__(self):
        super().__init__()

        self.all_predicted_types = []
        self.full_type_loss = tf.constant(0.0)

    def eval_full_loss(self, status):
        super().eval_full_loss(status)

        with open(Configuration.cooperative__compared_types) as type_result_file:
            type_result = type_result_file.readlines()

        for prob, result in zip(self.all_predicted_types, type_result):
            if result == "true":
                self.full_type_loss = self.full_type_loss - tf.math.log(prob + self.MIN_LG)
            elif result == "false":
                self.full_type_loss = self.full_type_loss - tf.math.log(1.0 - prob + self.MIN_LG)
        # full_type_loss = full_type_loss / tf.constant(len(all_predicted_types), dtype='float32')

    def print_loss(self):
        super().print_loss()
        print('types : %.4f' % self.full_type_loss.numpy())

    def get_full_loss(self):
        return super().get_full_loss() + self.full_type_loss
