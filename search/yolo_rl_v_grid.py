import clify
import argparse
import numpy as np

from dps import cfg
from dps.config import DEFAULT_CONFIG

from dps.projects.nips_2018 import envs
from dps.projects.nips_2018.algs import yolo_rl_config as alg_config
from dps.train import PolynomialScheduleHook


parser = argparse.ArgumentParser()
parser.add_argument("kind", choices="long_cedar med_cedar short_cedar".split())
args, _ = parser.parse_known_args()
kind = args.kind


fragment = [
    dict(obj_exploration=0.2, preserve_env=False),
    dict(obj_exploration=0.1,),
    dict(obj_exploration=0.05,),
    dict(do_train=False, n_train=16, min_chars=1, postprocessing="", preserve_env=False),
]


distributions = dict(
    area_vs_nonzero=list(np.linspace(0.01, 0.99, 24)),
)


def stage_prepare_func(stage_idx):
    if cfg.fixed_values:  # First two stages.
        return

    from dps import cfg
    cfg.area_weight = cfg.area_vs_nonzero * cfg.cost_weight
    cfg.nonzero_weight = (1 - cfg.area_vs_nonzero) * cfg.cost_weight


config = DEFAULT_CONFIG.copy()

config.update(alg_config)

config.update(envs.grid_config)

config.update(
    render_step=100000,
    eval_step=1000,
    per_process_gpu_memory_fraction=0.3,

    max_experiences=100000000,
    patience=2500,
    max_steps=100000000,
    stopping_criteria="mAP,max",
    threshold=0.99,

    area_weight=None,
    nonzero_weight=None,
    area_vs_nonzero=None,
    cost_weight=1.0,

    curriculum=[
        dict(max_steps=5000, rl_weight=None, stopping_criteria="TOTAL_COST,min", threshold=-np.inf,
             fixed_values=dict(h=8/14, w=8/14, cell_x=0.5, cell_y=0.5, obj=1.0)),
        dict(max_steps=5000, rl_weight=None, stopping_criteria="TOTAL_COST,min", threshold=-np.inf,
             fixed_values=dict(obj=1.0)),
    ],

    hooks=[
        PolynomialScheduleHook(
            attr_name="cost_weight",
            query_name="best_COST_reconstruction",
            base_configs=fragment, tolerance=10,
            initial_value=10, scale=10, power=1.0)
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
        max_hosts=2, ppn=12, cpp=2, gpu_set="0,1,2,3", wall_time="24hours", project="rrg-dprecup",
        cleanup_time="30mins", slack_time="30mins")

elif kind == "med_cedar":
    kind_args = dict(
        max_hosts=1, ppn=6, cpp=1, gpu_set="0,1", wall_time="1hour", project="rrg-dprecup",
        cleanup_time="10mins", slack_time="10mins", n_param_settings=6)

elif kind == "short_cedar":
    kind_args = dict(
        max_hosts=1, ppn=3, cpp=1, gpu_set="0", wall_time="20mins", project="rrg-dprecup",
        cleanup_time="2mins", slack_time="2mins", n_param_settings=3)

else:
    raise Exception("Unknown kind: {}".format(kind))

run_kwargs.update(kind_args)

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(
    name="grid_param_search_{}".format(kind), config=config,
    distributions=distributions, **run_kwargs)
