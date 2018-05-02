import clify
import argparse
import numpy as np

from dps.config import DEFAULT_CONFIG

from dps.projects.nips_2018 import envs
from dps.projects.nips_2018.algs import yolo_rl_config as alg_config


parser = argparse.ArgumentParser()
parser.add_argument("kind", choices="long_cedar med_cedar short_cedar".split())
args, _ = parser.parse_known_args()
kind = args.kind


distributions = dict(
    area_weight=[1, 2, 4, 8, 16, 32],
    nonzero_weight=[16, 32, 64, 12],
    z_weight=[None, 1],
)


config = DEFAULT_CONFIG.copy()

config.update(alg_config)

config.update(envs.grid_config)

config.update(
    render_step=100000,
    eval_step=1000,
    per_process_gpu_memory_fraction=0.3,

    patience=10000,
    max_experiences=100000000,
    max_steps=100000000,
    stopping_criteria="TOTAL_COST,min",
    threshold=-np.inf,

    hw_weight=None,
    rl_weight=1.0,

    curriculum=[
        dict(obj_exploration=0.3,
             load_path="/scratch/e2crawfo/dps_data/logs/yolo_rl_VERSUS_nips_2018_grid/reference_point/weights/best_of_stage_1",),
        dict(obj_exploration=0.2),
        dict(obj_exploration=0.1),
        dict(obj_exploration=0.05),
        dict(do_train=False, n_train=16, min_chars=1, postprocessing="", preserve_env=False),
        dict(postprocessing="", preserve_env=False),
    ]
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
        max_hosts=2, ppn=12, cpp=2, gpu_set="0,1,2,3", wall_time="12hours", project="def-jpineau",
        cleanup_time="30mins", slack_time="30mins", n_param_settings=24)

elif kind == "med_cedar":
    kind_args = dict(
        max_hosts=1, ppn=6, cpp=1, gpu_set="0,1", wall_time="1hour", project="def-jpineau",
        cleanup_time="10mins", slack_time="10mins", n_param_settings=6)

elif kind == "short_cedar":
    kind_args = dict(
        max_hosts=1, ppn=3, cpp=1, gpu_set="0", wall_time="20mins", project="def-jpineau",
        cleanup_time="2mins", slack_time="2mins", n_param_settings=3)

else:
    raise Exception("Unknown kind: {}".format(kind))

run_kwargs.update(kind_args)

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(
    name="grid_param_search_{}".format(kind), config=config,
    distributions=distributions, **run_kwargs)
