import tensorflow as tf

from dps import cfg
from dps.utils import Param
from dps.datasets.atari import RewardClassificationDataset

from auto_yolo.models import yolo_math


class Env(object):
    def __init__(self):
        train = RewardClassificationDataset(
            rl_data_location=cfg.train_rl_data_location,
            max_examples_per_class=cfg.n_train_per_class,
            balanced=cfg.train_balanced)

        val = RewardClassificationDataset(
            rl_data_location=cfg.val_rl_data_location,
            max_examples_per_class=cfg.n_val_per_class,
            balanced=True)

        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


class YoloAIR_XONetwork(yolo_math.YoloAir_MathNetwork):
    largest_digit = None
    n_actions = Param()
    classes = Param()

    def __init__(self, env, **kwargs):
        super(YoloAIR_XONetwork, self).__init__(env, **kwargs)
        self.eval_funcs = dict()

    @property
    def n_classes(self):
        return len(self.classes)

    def build_math_representation(self, math_attr):
        one_hot_actions = tf.one_hot(tf.to_int32(self._tensors["actions"][:, 0]), self.n_actions)
        actions = tf.tile(one_hot_actions[:, None, None, None, :], (1, self.H, self.W, self.B, 1))

        attr_rep = self._tensors["raw_obj"] * math_attr

        return tf.concat([attr_rep, self.program["box"], actions], axis=4)

    def _process_labels(self, labels):
        self._tensors.update(
            actions=labels[0],
            targets=labels[1],
        )


class SimpleXONetwork(yolo_math.SimpleMathNetwork):
    largest_digit = None
    n_actions = Param()
    classes = Param()

    @property
    def n_classes(self):
        return len(self.classes)

    def build_math_representation(self, math_code):
        one_hot_actions = tf.one_hot(tf.to_int32(self._tensors["actions"][:, 0]), self.n_actions)
        actions = tf.tile(one_hot_actions[:, None, None, None, :], (1, self.H, self.W, 1, 1))
        return tf.concat([math_code, actions], axis=4)

    def _process_labels(self, labels):
        self._tensors.update(
            actions=labels[0],
            targets=labels[1],
        )
