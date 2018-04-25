from dps import cfg
from dps.datasets import GridEmnistObjectDetection
from dps.utils import Config


class Nips2018Grid(object):
    def __init__(self):
        train = GridEmnistObjectDetection(n_examples=int(cfg.n_train), shuffle=True, example_range=(0.0, 0.9))
        val = GridEmnistObjectDetection(n_examples=int(cfg.n_val), shuffle=True, example_range=(0.9, 1.))

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
    n_sub_image_examples=0,
    image_shape=(6*14, 6*14),
    sub_image_shape=(14, 14),
    characters=list(range(10)),
    sub_image_size_std=0.0,
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
    postprocessing="",
    max_time_steps=25,
)

if __name__ == "__main__":
    with grid_config.copy(n_train=100, n_val=100):
        obj = Nips2018Grid()
        obj.datasets['train'].visualize()
