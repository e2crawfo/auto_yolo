from dps import cfg
from dps.datasets import EmnistObjectDetection
from dps.utils import Config


class Nips2018Scattered(object):
    def __init__(self):
        train = EmnistObjectDetection(n_examples=int(cfg.n_train), shuffle=True, example_range=(0.0, 0.9))
        val = EmnistObjectDetection(n_examples=int(cfg.n_val), shuffle=True, example_range=(0.9, 1.))

        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


test_curric=[dict(min_chars=nc, max_chars=nc, do_train=False) for nc in range(1, 16)]


config = Config(
    log_name="nips_2018_scattered",
    build_env=Nips2018Scattered,

    # dataset params
    use_dataset_cache=True,
    min_chars=10,
    max_chars=10,
    n_sub_image_examples=0,
    image_shape=(5*14, 5*14),
    sub_image_shape=(14, 14),
    characters=list(range(10)),
    colours="white",
    max_overlap=400,

    xent_loss=True,

    n_train=1e5,
    n_val=2**7,
    n_test=2**7,
)

config_train_5 = config.copy(
    min_chars=5,
    max_chars=5,
)

config_train_10 = config.copy(
    min_chars=10,
    max_chars=10,
)

config_train_15 = config.copy(
    min_chars=15,
    max_chars=15,
)
