import clify
import argparse
import numpy as np

from dps.config import DEFAULT_CONFIG

from dps.projects.nips_2018 import envs
from dps.projects.nips_2018.algs import yolo_rl_config as alg_config


parser = argparse.ArgumentParser()
parser.add_argument("kind", choices="long_cedar long_graham med short".split())
args, _ = parser.parse_known_args()
kind = args.kind


distributions = dict(
    area_weight=[1, 2, 3, 4, 5],
    nonzero_weight=[10, 20, 30, 40, 50],
)


config = DEFAULT_CONFIG.copy()

config.update(alg_config)

config.update(envs.grid_config)

config.update(
    render_step=100000,
    eval_step=1000,
    per_process_gpu_memory_fraction=0.23,

    patience=10000,
    max_experiences=100000000,
    max_steps=100000000,
    stopping_criteria="TOTAL_COST,min",
    threshold=-np.inf,
)

config.log_name = "{}_VS_{}".format(alg_config.log_name, envs.grid_config.log_name)
run_kwargs = dict(
    n_repeats=1,
    kind="slurm",
    pmem=5000,
    ignore_gpu=False,
)

if kind == "long_cedar":
    kind_args = dict(
        max_hosts=2, ppn=12, cpp=2, gpu_set="0,1,2,3", wall_time="11hours", project="def-jpineau",
        cleanup_time="30mins", slack_time="30mins", n_param_settings=24)

if kind == "long_graham":
    kind_args = dict(
        max_hosts=3, ppn=8, cpp=2, gpu_set="0,1", wall_time="12hours", project="def-jpineau",
        cleanup_time="30mins", slack_time="30mins", n_param_settings=24)

elif kind == "med":
    kind_args = dict(
        max_hosts=1, ppn=4, cpp=2, gpu_set="0", wall_time="1hour", project="def-jpineau",
        cleanup_time="10mins", slack_time="10mins", n_param_settings=4)

elif kind == "short":
    kind_args = dict(
        max_hosts=1, ppn=4, cpp=1, gpu_set="0", wall_time="20mins", project="rpp-bengioy",
        cleanup_time="5mins", slack_time="5mins", n_param_settings=4)

else:
    raise Exception("Unknown kind: {}".format(kind))

run_kwargs.update(kind_args)

readme = "Testing new yolo_rl on grid task, trying to get centered again after making some changes that I probably shouldn't have done."

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(
    name="grid_param_search_{}".format(kind), config=config, readme=readme,
    distributions=distributions, **run_kwargs)
