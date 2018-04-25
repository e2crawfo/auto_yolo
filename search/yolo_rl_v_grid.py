import clify
import argparse
import numpy as np

from dps import cfg
from dps.config import DEFAULT_CONFIG

from dps.projects.nips_2018.envs import grid_config as env_config
from dps.projects.nips_2018.algs import yolo_rl_config as alg_config
from dps.train import PolynomialScheduleHook


fragment = [
    dict(obj_exploration=0.2,),
    dict(obj_exploration=0.1,),
    dict(obj_exploration=0.05,),
    dict(do_train=False, n_train=16, min_chars=1, postprocessing="", preserve_env=False),
]


distributions = dict(
    area_weight=list(np.linspace(0.1, 1.0, 24))
)

config = DEFAULT_CONFIG.copy()

config.update(alg_config)
config.update(env_config)

config.update(
    render_step=100000,
    eval_step=1000,
    per_process_gpu_memory_fraction=0.3,

    # max_experiences=100000,
    patience=5000,
    max_steps=10000,

    area_weight=None,
    nonzero_weight=None,

    curriculum=[
        dict(do_train=False),
    ],

    hooks=[
        PolynomialScheduleHook(
            attr_name="nonzero_weight",
            query_name="best_COST_reconstruction",
            base_configs=fragment, tolerance=10,
            initial_value=30,
            scale=10, power=1.0)
    ]
)

config.log_name = "{}_VS_{}".format(alg_config.log_name, env_config.log_name)

print("Forcing creation of first dataset.")
with config.copy():
    cfg.build_env()

print("Forcing creation of second dataset.")
with config.copy(config.curriculum[-1]):
    cfg.build_env()

run_kwargs = dict(
    n_repeats=1,
    kind="slurm",
    pmem=5000,
    ignore_gpu=False,
)

parser = argparse.ArgumentParser()
parser.add_argument("kind", choices="long_cedar long_graham short_graham short_cedar".split())
args, _ = parser.parse_known_args()
kind = args.kind

if kind == "long_cedar":
    kind_args = dict(
        max_hosts=2, ppn=12, cpp=1, gpu_set="0,1,2,3", wall_time="6hours", project="rrg-dprecup",
        cleanup_time="30mins", slack_time="30mins", n_param_settings=24,)

elif kind == "long_graham":
    kind_args = dict(
        max_hosts=2, ppn=8, cpp=1, gpu_set="0,1", wall_time="6hours", project="def-jpineau",
        cleanup_time="30mins", slack_time="30mins", n_param_settings=16)

elif kind == "short_cedar":
    kind_args = dict(
        max_hosts=1, ppn=3, cpp=1, gpu_set="0", wall_time="20mins", project="rrg-dprecup",
        cleanup_time="2mins", slack_time="2mins", n_param_settings=3)

elif kind == "short_graham":
    kind_args = dict(
        max_hosts=1, ppn=4, cpp=1, gpu_set="0", wall_time="20mins", project="def-jpineau",
        cleanup_time="2mins", slack_time="2mins", n_param_settings=4)

else:
    raise Exception("Unknown kind: {}".format(kind))

run_kwargs.update(kind_args)

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(
    name="grid_task_param_search_{}".format(kind), config=config,
    distributions=distributions, **run_kwargs)
