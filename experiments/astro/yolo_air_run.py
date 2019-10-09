from astropy.utils.data import get_pkg_data_filename

from auto_yolo import algs
from auto_yolo.datasets.fits import FITSDataset

from dps import cfg
from dps.datasets.base import Environment
from dps.hyper.base import run_experiment
from dps.utils import Config

readme = "Running YOLO AIR experiment on astro data."

alg_config = algs.yolo_air_conv_config


class AstroEnv(Environment):
    def __init__(self):
        train = FITSDataset(fits_file=cfg.train_file)
        val = FITSDataset(fits_file=cfg.val_file)
        test = FITSDataset(fits_file=cfg.test_file)

        self.datasets = dict(
            train=train, val=val, test=test,
        )

    def close(self):
        pass


temp_image_file = get_pkg_data_filename('galactic_center/gc_2mass_k.fits')


env_config = dict(
    env_name='astro',
    build_env=AstroEnv,
    train_file=temp_image_file,
    val_file=temp_image_file,
    test_file=temp_image_file,
    force_memmap=False,
    postprocessing='random',
    tile_shape=(50, 50),
    n_samples_per_image=100,

    background_cfg=dict(mode="colour", colour="black"),
    object_shape=(14, 14),
)

config = Config(alg_config, **env_config)

distributions = None
durations = dict()

run_experiment(
    "yolo_air_vs_astro", config, readme, distributions=distributions, durations=durations,
)
