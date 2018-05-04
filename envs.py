from dps import cfg
from dps.datasets import GridEmnistObjectDetectionDataset, EmnistObjectDetectionDataset
from dps.utils import Config, gen_seed
from dps.env.advanced import yolo_rl


class Nips2018Grid(object):
    def __init__(self):
        train = GridEmnistObjectDetectionDataset(n_examples=int(cfg.n_train), shuffle=True, example_range=(0.0, 0.9), seed=gen_seed())
        val = GridEmnistObjectDetectionDataset(n_examples=int(cfg.n_val), shuffle=True, example_range=(0.9, 1.), seed=gen_seed())

        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


class Nips2018Scatter(object):
    def __init__(self):
        train = EmnistObjectDetectionDataset(n_examples=int(cfg.n_train), shuffle=True, example_range=(0.0, 0.9), seed=gen_seed())
        val = EmnistObjectDetectionDataset(n_examples=int(cfg.n_val), shuffle=True, example_range=(0.9, 1.), seed=gen_seed())

        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


grid_config = Config(
    log_name="nips_2018_grid",
    build_env=Nips2018Grid,
    seed=347405995,

    # dataset params
    min_chars=16,
    max_chars=25,
    n_patch_examples=0,
    image_shape=(6*14, 6*14),
    patch_shape=(14, 14),
    characters=list(range(10)),
    # characters=list(range(10)) + "A M X N".split(),
    patch_size_std=0.0,
    colours="white",

    grid_shape=(6, 6),
    spacing=(-3, -3),
    random_offset_range=(15, 15),

    n_distractors_per_image=0,

    backgrounds="",
    backgrounds_sample_every=False,
    background_colours="",

    background_cfg=dict(mode="none"),

    object_shape=(14, 14),

    xent_loss=True,

    postprocessing="random",
    n_samples_per_image=4,
    tile_shape=(32, 32),
    preserve_env=True,

    n_train=25000,
    n_val=1e2,
    n_test=1e2,

    eval_step=1000,
    display_step=1000,
    render_step=5000,
    max_steps=1e7,
    patience=10000,
)

grid_fullsize_config = grid_config.copy(
    log_name="nips_2018_gridfullsize",
    postprocessing="",
    max_time_steps=16,
    max_chars=16,
    min_chars=16,
    grid_shape=(4, 4),
    spacing=(0, 0),
    random_offset_range=None,
    image_shape=(56, 56),

)

air_testing_config = grid_config.copy(
    log_name="nips_2018_air_testing",
    postprocessing="",
    max_time_steps=4,
    max_chars=4,
    min_chars=4,
    grid_shape=(2, 2),
    spacing=(0, 0),
    random_offset_range=None,
    image_shape=(28, 28),

)


scatter_white_14x14_config = grid_config.copy(
    log_name="nips_2018_scatter_white",
    build_env=Nips2018Scatter,
    max_overlap=196/2,
    min_chars=15,
    max_chars=15,
    tile_shape=(40, 40),
)

scatter_colour_14x14_config = scatter_white_14x14_config.copy(
    log_name="nips_2018_scatter_colour",
    build_env=Nips2018Scatter,
    colours="red blue green cyan yellow magenta",
)

scatter_white_28x28_config = grid_config.copy(
    log_name="nips_2018_scatter_white_28x28",
    build_env=Nips2018Scatter,
    object_shape=(28, 28),
    colours="white",
    min_chars=1,
    max_chars=15,
    max_overlap=2*196,
    patch_shape=(28, 28),
    tile_shape=(48, 48),
    image_shape=(100, 100),
    build_object_decoder=yolo_rl.ObjectDecoder28x28,
)

scatter_colour_28x28_config = scatter_white_28x28_config.copy(
    log_name="nips_2018_scatter_colour_28x28",
    colours="red blue green cyan yellow magenta",
    build_object_decoder=yolo_rl.ObjectDecoder28x28,
)

single_digit_config = grid_config.copy(
    log_name="nips_2018_single_digit",
    build_env=Nips2018Scatter,
    min_chars=1,
    max_chars=1,
    image_shape=(32, 32),
    fixed_values=dict(alpha=1.0),

    postprocessing="",
)


double_digit_config = grid_config.copy(
    log_name="nips_2018_double_digit",
    build_env=Nips2018Scatter,
    min_chars=1,
    max_chars=2,
    image_shape=(32, 32),
    min_yx=-1.0,
    max_yx=2.0,
    box_std=0.1,
    max_overlap=196/2,

    postprocessing="",

    area_weight="Poly(0.01, 10.0, 100000)",
    nonzero_weight=10.0,
    max_steps=100000,
    curriculum=[
        dict(obj_exploration=0.2),
        # dict(obj_exploration=0.1),
        # dict(obj_exploration=0.05),
        dict(do_train=False, n_train=16, min_chars=1, postprocessing="", preserve_env=False),
        dict(obj_exploration=0.05, preserve_env=False, patience=10000000),
    ],
)


if __name__ == "__main__":
    with scatter_colour_14x14_config.copy(n_train=100, n_val=100):
        obj = Nips2018Scatter()
