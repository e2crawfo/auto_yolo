import argparse

from dps import cfg
from dps.datasets import (
    GridEmnistObjectDetectionDataset, EmnistObjectDetectionDataset,
    VisualArithmeticDataset, GameDataset)
from dps.env.basic import collect
from dps.datasets.shapes import ShapesDataset
from dps.datasets.clevr import ClevrDataset
from dps.utils import Config, pdb_postmortem
from dps.train import training_loop
from dps.hyper import build_and_submit
from dps.config import DEFAULT_CONFIG

import auto_yolo.algs as alg_module


def sanitize(s):
    return s.replace('_', '-')


def run_experiment(
        name, config, readme, distributions=None, durations=None,
        alg=None, task="grid", name_variables=None, env_kwargs=None):

    name = sanitize(name)
    durations = durations or {}

    parser = argparse.ArgumentParser()
    parser.add_argument("duration", choices=list(durations.keys()) + ["local"])
    parser.add_argument('--pdb', action='store_true',
                        help="If supplied, enter post-mortem debugging on error.")

    args, _ = parser.parse_known_args()

    _config = DEFAULT_CONFIG.copy()

    env_kwargs = env_kwargs or {}

    env_kwargs['task'] = task
    env_config = get_env_config(**env_kwargs)
    _config.update(env_config)

    if alg:
        alg_config = getattr(alg_module, "{}_config".format(alg))
        _config.update(alg_config)
        alg_name = sanitize(alg_config.alg_name)
    else:
        alg_name = ""

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
        train_seed, val_seed, test_seed = 0, 1, 2
        train = GridEmnistObjectDetectionDataset(
            n_examples=int(cfg.n_train), shuffle=True,
            example_range=cfg.train_example_range, seed=train_seed)

        val = GridEmnistObjectDetectionDataset(
            n_examples=int(cfg.n_val), shuffle=True,
            example_range=cfg.val_example_range, seed=val_seed)

        test = GridEmnistObjectDetectionDataset(
            n_examples=int(cfg.n_val), shuffle=True,
            example_range=cfg.test_example_range, seed=test_seed)

        self.datasets = dict(train=train, val=val, test=test)

    def close(self):
        pass


class Nips2018Scatter(object):
    def __init__(self):
        train_seed, val_seed, test_seed = 0, 1, 2
        train = EmnistObjectDetectionDataset(
            n_examples=int(cfg.n_train), shuffle=True,
            example_range=cfg.train_example_range, seed=train_seed)

        val = EmnistObjectDetectionDataset(
            n_examples=int(cfg.n_val), shuffle=True,
            example_range=cfg.val_example_range, seed=val_seed)

        test = EmnistObjectDetectionDataset(
            n_examples=int(cfg.n_val), shuffle=True,
            example_range=cfg.test_example_range, seed=test_seed)

        self.datasets = dict(train=train, val=val, test=test)

    def close(self):
        pass


class Nips2018Arithmetic(object):
    def __init__(self):
        train_seed, val_seed, test_seed = 0, 1, 2

        train = VisualArithmeticDataset(
            n_examples=int(cfg.n_train), shuffle=True,
            example_range=cfg.train_example_range, seed=train_seed)

        val = VisualArithmeticDataset(
            n_examples=int(cfg.n_val), shuffle=True,
            example_range=cfg.val_example_range, seed=val_seed)

        test = VisualArithmeticDataset(
            n_examples=int(cfg.n_val), shuffle=True,
            example_range=cfg.test_example_range, seed=test_seed)

        self.datasets = dict(train=train, val=val, test=test)

    def close(self):
        pass


class Nips2018Shapes(object):
    def __init__(self):
        train_seed, val_seed, test_seed = 0, 1, 2

        train = ShapesDataset(n_examples=cfg.n_train, seed=train_seed)
        val = ShapesDataset(n_examples=cfg.n_val, seed=val_seed)
        test = ShapesDataset(n_examples=cfg.n_val, seed=test_seed)

        self.datasets = dict(train=train, val=val, test=test)

    def close(self):
        pass


class Nips2018Clevr(object):
    def __init__(self):
        train_seed, val_seed, test_seed = 0, 1, 2

        train = ClevrDataset(clevr_kind="train", n_examples=cfg.n_train, seed=train_seed, example_range=None)
        val = ClevrDataset(clevr_kind="val", n_examples=cfg.n_val, seed=val_seed, example_range=cfg.val_example_range)
        test = ClevrDataset(clevr_kind="val", n_examples=cfg.n_val, seed=test_seed, example_range=cfg.test_example_range)

        self.datasets = dict(train=train, val=val, test=test)

    def close(self):
        pass


class Nips2018Collect(object):
    def __init__(self):
        train_seed, val_seed, test_seed = 0, 1, 2

        env = collect.build_env().gym_env
        train = GameDataset(env=env, n_examples=cfg.n_train, seed=train_seed)
        val = GameDataset(env=env, n_examples=cfg.n_val, seed=val_seed)
        test = GameDataset(env=env, n_examples=cfg.n_val, seed=test_seed)

        self.datasets = dict(train=train, val=val, test=test)

    def close(self):
        pass


env_config = Config(
    train_example_range=(0.0, 0.8),
    val_example_range=(0.8, 0.9),
    test_example_range=(0.9, 1.0),
)


grid_config = env_config.copy(
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
    n_val=1e3,

    eval_step=1000,
    display_step=1000,
    render_step=5000,
    patience=1000000,
    max_steps=110000,
)


def get_env_config(task, size=14, in_colour=False, ops="addition", image_size="normal", **_):
    if task == "xo":
        return env_config.copy(
            env_name="xo",
            # build_env=Nips2018XO,
            one_hot=True,

            image_shape=(72, 72),
            postprocessing="",

            max_entities=30,
            max_episode_length=100,

            n_train=1000,
            n_val=1000,

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
    elif task == "collect":
        config = env_config.copy(collect.config)
        config.render_hook = None
        config.hooks = []
        config.exploration_schedule = None
        config.update(
            env_name="collect", build_env=Nips2018Collect,
            n_train=25000, n_val=1000, keep_prob=0.25,
            background_cfg=dict(mode="colour", colour="white"))
        return config

    elif task == "collect_rl":
        config = env_config.copy(collect.config)
        return config

    config = grid_config.copy()
    config.env_name = "task={}".format(task)

    if task == "arithmetic":
        config.env_name += "_ops={}".format(ops)

    if in_colour:
        config.colours = "red blue green cyan yellow magenta"

    if task == "grid":
        if image_size == "small":
            config.update(
                min_chars=1,
                max_chars=4,
                image_shape=(50, 50),
                grid_shape=(3, 3),
                spacing=(0, 0),
                random_offset_range=(8, 8),
                colours="",
            )
        elif image_size == "pretrain":
            config.update(
                min_chars=1,
                max_chars=1,
                image_shape=(15, 15),
                grid_shape=(1, 1),
                spacing=(-3, -3),
                random_offset_range=(1, 1),
            )

        return config

    size = int(size)

    if size == 14:
        config.update(
            max_overlap=14*14/2,
            min_chars=15,
            max_chars=15,
        )
    elif size == 21:
        config.update(
            max_overlap=21*21/2,
            min_chars=12,
            max_chars=12,
            patch_size_std=0.05,
        )
    else:
        raise Exception("Unknown size {}".format(size))

    if task == "scatter":
        config.build_env = Nips2018Scatter

    elif task == "arithmetic":
        config.update(
            build_env=Nips2018Arithmetic,
            image_shape=(48, 48),

            min_digits=1,

            max_digits=9,
            n_classes=82,
            largest_digit=81,

            # max_digits=6,
            # n_classes=55,
            # largest_digit=54,

            one_hot=True,
            reductions="sum" if ops == "addition" else "A:sum,N:min,X:max,C:len",
        )
    elif task == "arithmetic2":
        config.update(
            build_env=Nips2018Arithmetic,
            image_shape=(48, 48),

            min_digits=5,

            max_digits=5,
            n_classes=46,
            largest_digit=45,

            # max_digits=6,
            # n_classes=55,
            # largest_digit=54,

            one_hot=True,
            reductions="sum" if ops == "addition" else "A:sum,N:min,X:max,C:len",
        )
    elif task == "small":
        config.update(
            build_env=Nips2018Arithmetic,
            image_shape=(24, 24),

            min_digits=1,

            max_digits=1,
            n_classes=10,
            largest_digit=9,

            # max_digits=6,
            # n_classes=55,
            # largest_digit=54,

            one_hot=True,
            reductions="sum" if ops == "addition" else "A:sum,N:min,X:max,C:len",
        )
    elif task == "shapes":
        config.update(
            build_env=Nips2018Shapes,
            image_shape=(48, 48),
            shapes="green,circle blue,circle orange,circle teal,circle red,circle black,circle",
            background_colours="white",
        )
    elif task == "clevr":
        config.update(
            build_env=Nips2018Clevr, image_shape=(80, 120),
        )
    else:
        raise Exception("Unknown task `{}`".format(task))
    return config
