import clify
import argparse
import numpy as np

from dps.config import DEFAULT_CONFIG

from dps.projects.nips_2018 import envs
from dps.projects.nips_2018.algs import yolo_math_config as alg_config


parser = argparse.ArgumentParser()
parser.add_argument("kind", choices="long med".split())
parser.add_argument("size", choices="14 21".split())
parser.add_argument("task", choices="addition arithmetic".split())
parser.add_argument("--c", action="store_true")
args, _ = parser.parse_known_args()
kind = args.kind


distributions = dict(
    math_weight=list(2**np.linspace(-2, 2, 8)),
)

env_config = envs.get_mnist_config(size=args.size, colour=args.c, task=args.task)

config = DEFAULT_CONFIG.copy()
config.update(alg_config)
config.update(env_config)
config.update(
    render_step=10000,
    eval_step=1000,
    per_process_gpu_memory_fraction=0.23,

    patience=1000000,
    max_experiences=100000000,
    max_steps=100000000,

    count_prior_decay_steps=1000,
    final_count_prior_log_odds=0.0125,
    hw_prior_std=0.5,
    kernel_size=1,
)

config.log_name = "parameter_search-{}_alg={}".format(env_config.log_name, alg_config.log_name)
run_kwargs = dict(
    n_repeats=1,
    kind="slurm",
    pmem=5000,
    ignore_gpu=False,
)

if kind == "long":
    kind_args = dict(
        max_hosts=1, ppn=8, cpp=2, gpu_set="0,1", wall_time="6hours", project="rpp-bengioy",
        cleanup_time="30mins", slack_time="30mins")

elif kind == "med":
    kind_args = dict(
        max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", wall_time="1hour", project="rpp-bengioy",
        cleanup_time="10mins", slack_time="10mins", n_param_settings=3, n_repeats=2)

else:
    raise Exception("Unknown kind: {}".format(kind))

run_kwargs.update(kind_args)

readme = "Parameter search on arithmetic task"

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(
    name=config.log_name + "_kind={}".format(kind),
    config=config, readme=readme, distributions=distributions, **run_kwargs)
