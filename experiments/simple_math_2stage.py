import clify
import argparse

from dps import cfg
from dps.config import DEFAULT_CONFIG
from dps.train import training_loop
from dps.projects.nips_2018 import envs
from dps.projects.nips_2018.algs import yolo_math_simple_config as alg_config

parser = argparse.ArgumentParser()
parser.add_argument("duration", choices="long med test".split())
parser.add_argument("size", choices="14 21".split())
parser.add_argument("task", choices="addition arithmetic".split())
parser.add_argument("--c", action="store_true")
args, _ = parser.parse_known_args()
duration = args.duration


distributions = dict(
    n_train=[1000, 2000, 4000, 8000, 16000, 32000],
)

env_config = envs.get_mnist_config(size=args.size, colour=args.c, task=args.task)


config = DEFAULT_CONFIG.copy()

config.update(alg_config)
config.update(env_config)

config.update(
    render_step=5000,
    eval_step=1000,
    per_process_gpu_memory_fraction=0.3,

    patience=5000,
    max_experiences=100000000,
    max_steps=110000,
    robust=False,

    variational=False,

    curriculum=[
        dict(
            math_weight=None,
            fixed_weights="math",
            stopping_criteria="loss_reconstruction,min",
            threshold=0.0,
            noise_schedule=0.001,  # Helps avoid collapse
        ),
        dict(
            max_steps=100000000,
            postprocessing="",
            preserve_env=False,
            math_weight=1.0,
            train_kl=False,
            train_reconstruction=False,
            fixed_weights="decoder encoder",
        )
    ],
)

config.log_name = "sample_complexity-{}_alg={}_2stage".format(env_config.log_name, alg_config.log_name)

run_kwargs = dict(
    n_repeats=6,
    kind="slurm",
    pmem=5000,
    ignore_gpu=False,
)

if duration == "long":
    duration_args = dict(
        max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", wall_time="6hours", project="rpp-bengioy",
        step_time_limit="2hours", cleanup_time="10mins", slack_time="5mins")

elif duration == "med":
    config.max_steps=1000
    duration_args = dict(
        max_hosts=1, ppn=3, cpp=2, gpu_set="0", wall_time="1hour", project="rpp-bengioy",
        cleanup_time="10mins", slack_time="3mins", n_param_settings=6, n_repeats=1, step_time_limit="25mins")

elif duration == "test":
    with config:
        cfg.update_from_command_line()
        training_loop()
else:
    raise Exception("Unknown duration: {}".format(duration))

run_kwargs.update(duration_args)

readme = "Running sample complexity experiment on {} task with simple_math network, double stage.".format(args.task)

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(
    name=config.log_name + "_duration={}".format(duration),
    config=config, readme=readme, distributions=distributions, **run_kwargs)
