import tensorflow as tf

from dps import cfg
from dps.utils import Param, Config
from dps.env.advanced import yolo_math, yolo_rl
from dps.datasets.atari import RewardClassificationDataset


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


env_config = Config(
    log_name="yolo_xo",
    build_env=Env,
    one_hot=True,

    # image_shape=(72, 72),
    # tile_shape=(48, 48),
    # postprocessing="random",
    # train_rl_data_location="/media/data/Dropbox/projects/PyDSRL/train",
    # val_rl_data_location="/media/data/Dropbox/projects/PyDSRL/val",
    # n_train_per_class=10000,

    n_train_per_class=5000,
    train_rl_data_location="/media/data/Dropbox/projects/PyDSRL/train_48x48",
    val_rl_data_location="/media/data/Dropbox/projects/PyDSRL/val_48x48",
    image_shape=(48, 48),
    tile_shape=(48, 48),
    postprocessing="",
    n_samples_per_image=4,
    train_balanced=False,

    classes=[-1, 0, 1],
    n_actions=4,

    n_val_per_class=100,
)

alg_config = Config(
    get_updater=yolo_rl.YoloRL_Updater,
    build_network=YoloAIR_XONetwork,

    math_weight=1.0,
    train_kl=True,
    train_reconstruction=True,
    min_yx=-0.5,
    max_yx=1.5,

    # build_math_network=yolo_math.SequentialRegressionNetwork,
    build_math_network=yolo_math.ObjectBasedRegressionNetwork,
    n_objects=3,

    curriculum=[
        dict(
            max_steps=3000,
            math_weight=0.0,
            fixed_weights="math",
            stopping_criteria="loss_reconstruction,min",
            threshold=0.0,
        ),
        dict(
            render_step=10000,
            max_steps=100000000,
            postprocessing="",
            preserve_env=False,
            train_kl=False,
            train_reconstruction=False,
            fixed_weights="object_encoder object_decoder box obj backbone edge",
        )
    ],
)

config = yolo_math.config.copy()
config.update(alg_config)
config.update(env_config)

simple_xo_config = yolo_math.simple_config.copy()
simple_xo_config.update(env_config)
simple_xo_config.update(
    build_network=SimpleXONetwork,
    train_reconstruction=False,
    train_kl=False,
    variational=False,
)

simple_xo_2stage_config = simple_xo_config.copy(
    curriculum=[
        dict(
            train_reconstruction=True,
            math_weight=0.0,
            fixed_weights="math",
            stopping_criteria="loss_reconstruction,min",
            threshold=0.0,
            noise_schedule=0.001,  # Helps avoid collapse
        ),
        dict(
            max_steps=100000000,
            postprocessing="",
            preserve_env=False,
            fixed_weights="object_encoder object_decoder box obj backbone edge",
        )
    ],
)
