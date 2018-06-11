import argparse

from dps import cfg
from dps.datasets import (
    GridEmnistObjectDetectionDataset, EmnistObjectDetectionDataset,
    VisualArithmeticDataset)
from dps.datasets.xo import XO_RewardClassificationDataset
from dps.utils import Config, pdb_postmortem
from dps.train import training_loop
from dps.hyper import build_and_submit
from dps.config import DEFAULT_CONFIG

import auto_yolo.algs as alg_module


def sanitize(s):
    return s.replace('_', '-')


def run_experiment(
        name, config, readme, distributions=None, durations=None,
        alg=None, task="grid", name_variables=None):

    name = sanitize(name)
    durations = durations or {}

    parser = argparse.ArgumentParser()
    parser.add_argument("duration", choices=list(durations.keys()) + ["local"])

    parser.add_argument('--alg', default=alg,
                        help="Name (or unique name-prefix) of algorithm to run. Optional. "
                             "If not provided, algorithm spec is assumed to be included "
                             "in the environment spec.")
    parser.add_argument("--size", choices="14 21".split(), default=14)
    parser.add_argument("--in-colour", action="store_true")
    parser.add_argument("--task", choices="grid scatter arithmetic xo".split(), default=task)
    parser.add_argument("--ops", choices="addition all".split(), default="addition")
    parser.add_argument('--pdb', action='store_true',
                        help="If supplied, enter post-mortem debugging on error.")

    args, _ = parser.parse_known_args()

    _config = DEFAULT_CONFIG.copy()

    env_kwargs = vars(args).copy()
    env_config = get_env_config(**env_kwargs)
    _config.update(env_config)

    alg_config = getattr(alg_module, "{}_config".format(args.alg))
    _config.update(alg_config)
    alg_name = sanitize(alg_config.alg_name)

    _config.update(config)
    _config.update_from_command_line()

    _config.env_name = "{}_env={}".format(name, sanitize(env_config.env_name))

    if args.duration == "local":
        _config.exp_name = "alg={}".format(alg_name)
        with _config:
            if args.pdb:
                with pdb_postmortem():
                    return training_loop()
            else:
                return training_loop()
    else:
        run_kwargs = Config(
            kind="slurm",
            pmem=5000,
            ignore_gpu=False,
        )

        duration_args = durations[args.duration]

        if 'config' in duration_args:
            _config.update(duration_args['config'])
            del duration_args['config']

        run_kwargs.update(durations[args.duration])
        run_kwargs.update_from_command_line()

    if name_variables is not None:
        name_variables_str = "_".join(
            "{}={}".format(sanitize(str(k)), sanitize(str(getattr(_config, k))))
            for k in name_variables.split(","))
        _config.env_name = "{}_{}".format(_config.env_name, name_variables_str)

    exp_name = "{}_alg={}_duration={}".format(_config.env_name, alg_name, args.duration)

    build_and_submit(
        name=exp_name, config=_config, distributions=distributions, **run_kwargs)


class Nips2018Grid(object):
    def __init__(self):
        train_seed, val_seed = 0, 1
        train = GridEmnistObjectDetectionDataset(
            n_examples=int(cfg.n_train), shuffle=True,
            example_range=(0.0, 0.9), seed=train_seed)

        val = GridEmnistObjectDetectionDataset(
            n_examples=int(cfg.n_val), shuffle=True,
            example_range=(0.9, 1.), seed=val_seed)

        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


class Nips2018Scatter(object):
    def __init__(self):
        train_seed, val_seed = 0, 1
        train = EmnistObjectDetectionDataset(
            n_examples=int(cfg.n_train), shuffle=True,
            example_range=(0.0, 0.9), seed=train_seed)

        val = EmnistObjectDetectionDataset(
            n_examples=int(cfg.n_val), shuffle=True,
            example_range=(0.9, 1.), seed=val_seed)

        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


class Nips2018Arithmetic(object):
    def __init__(self):
        train_seed, val_seed = 0, 1

        train = VisualArithmeticDataset(
            n_examples=int(cfg.n_train), shuffle=True,
            example_range=(0.0, 0.9), seed=train_seed)

        val = VisualArithmeticDataset(
            n_examples=int(cfg.n_val), shuffle=True,
            example_range=(0.9, 1.), seed=val_seed)

        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


class Nips2018XO(object):
    def __init__(self):
        train_seed, val_seed = 0, 1

        train = XO_RewardClassificationDataset(n_examples=cfg.n_train, seed=train_seed)

        val = XO_RewardClassificationDataset(n_examples=cfg.n_val, seed=val_seed)

        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


grid_config = Config(
    env_name="nips_2018_grid",
    build_env=Nips2018Grid,

    # dataset params
    min_chars=16,
    max_chars=25,
    n_patch_examples=0,
    image_shape=(6*14, 6*14),
    patch_shape=(14, 14),
    characters=list(range(10)),
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

    postprocessing="",
    preserve_env=False,

    n_train=25000,
    n_val=1e2,

    eval_step=1000,
    display_step=1000,
    render_step=5000,
    patience=1000000,
    max_steps=110000,

    overwrite_plots=False,
)

air_testing_config = grid_config.copy(
    env_name="nips_2018_air_testing",
    postprocessing="",
    max_time_steps=4,
    max_chars=4,
    min_chars=4,
    grid_shape=(2, 2),
    spacing=(0, 0),
    random_offset_range=None,
    image_shape=(28, 28),

)


def get_env_config(task, size, in_colour, ops, **_):
    if task == "xo":
        return Config(
            env_name="xo",
            build_env=Nips2018XO,
            one_hot=True,

            image_shape=(72, 72),
            postprocessing="",

            max_entities=30,
            max_episode_length=100,

            n_train=1000,
            n_val=100,

            balanced=True,
            classes=[-1, 0, 1],
            n_actions=4,

            backgrounds="",
            backgrounds_sample_every=False,
            background_colours="",
            background_cfg=dict(mode="none"),

            eval_step=1000,
            display_step=1000,
            render_step=5000,
        )

    config = grid_config.copy()
    config.env_name = "size={}_in-colour={}_task={}".format(size, in_colour, task)

    if task == "arithmetic":
        config.env_name += "_ops={}".format(ops)

    if in_colour:
        config.colours = "red blue green cyan yellow magenta"

    if task == "grid":
        return config

    size = int(size)

    if size == 14:
        config.update(
            max_overlap=14*14/2,
            min_chars=15,
            max_chars=15,
            tile_shape=(48, 48),
        )
    elif size == 21:
        config.update(
            max_overlap=21*21/2,
            min_chars=12,
            max_chars=12,
            tile_shape=(48, 48),
            patch_size_std=0.05,
        )
    else:
        raise Exception("Unknown size {}".format(size))

    if task == "scatter":
        config.build_env = Nips2018Scatter

    elif task == "arithmetic":
        config.update(
            build_env=Nips2018Arithmetic,
            min_digits=1,
            max_digits=11,

            largest_digit=99,
            one_hot=True,
            reductions="sum" if ops == "addition" else "A:sum,N:min,X:max,C:len",

            postprocessing="",
        )
    else:
        raise Exception("Unknown task `{}`".format(task))
    return config


def prepare_func():
    from dps import cfg
    cfg.curriculum[1]['nonzero_weight'] = "Poly(0.0, {}, 100000)".format(cfg.nonzero_weight)


# So this pretty much works!!
yolo_rl_double_digit_config = grid_config.copy(
    env_name="nips_2018_double_digit",
    build_env=Nips2018Scatter,
    min_chars=1,
    max_chars=2,
    image_shape=(48, 48),
    box_std=0.1,

    min_hw=0.25,
    max_hw=3.,
    pixels_per_cell=(12, 12),

    postprocessing="",
    patience=1000000,
    render_step=5000,

    # patch_shape=(21, 21),
    # max_overlap=196,
    # patch_shape=(14, 14),
    # max_overlap=196/4,
    patch_shape=(28, 28),
    max_overlap=196,

    min_yx=-0.5,
    max_yx=1.5,

    use_input_attention=True,
    order="obj z box",

    prepare_func=prepare_func,

    area_neighbourhood_size=2,
    hw_neighbourhood_size=None,  # This is what it was set to, but only by accident
    nonzero_neighbourhood_size=2,
    local_reconstruction_cost=True,

    area_weight=1.0,
    hw_weight=40.,
    nonzero_weight=30.,
    reconstruction_weight=1.0,
    max_steps=100000,

    curriculum=[
        {'hw_weight': None,
         'rl_weight': None,
         'obj_default': 0.5,
         'obj_exploration': 1.0},
        {'obj_exploration': 0.2},
        {'obj_exploration': 0.1},
        {'obj_exploration': 0.05},
        {'obj_exploration': 0.01}
    ],
    sequential_cfg=dict(on=True),
    n_passthrough_features=100,

    readme="Testing the standard set up, which we've determined should work fairly well."
)
