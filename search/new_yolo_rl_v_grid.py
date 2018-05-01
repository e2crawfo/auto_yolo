import clify
import argparse

from dps import cfg
from dps.config import DEFAULT_CONFIG

from dps.projects.nips_2018.envs import grid_config as env_config
from dps.projects.nips_2018.algs import yolo_rl_config as alg_config


distributions = dict(
    hook_scale=[50, 75, 150],
)


def prepare_func():
    from dps import cfg
    from dps.train import PolynomialScheduleHook

    fragment = [
        dict(obj_exploration=0.2, preserve_env=False),
        dict(obj_exploration=0.1),
        dict(obj_exploration=0.05),
        dict(do_train=False, n_train=16, min_chars=1, postprocessing="", preserve_env=False),
    ]

    cfg.hooks=[
        PolynomialScheduleHook(
            attr_name="nonzero_weight",
            query_name="best_COST_reconstruction",
            base_configs=fragment, tolerance=10,
            initial_value=50,
            scale=cfg.hook_scale, power=1.0)
    ]


config = DEFAULT_CONFIG.copy()

config.update(alg_config)
config.update(env_config)

config.update(
    prepare_func=prepare_func,
    render_step=100000,
    eval_step=1000,
    per_process_gpu_memory_fraction=0.3,

    # max_experiences=100000,
    max_experiences=100000000,
    patience=2500,
    max_steps=100000000,

    area_weight=0.8,
    nonzero_weight=30,

    curriculum=[
        dict(do_train=False),
    ],

    sequential_cfg=dict(on=True),
    hooks=[],
)

config.log_name = "{}_VS_{}".format(alg_config.log_name, env_config.log_name)

print("Forcing creation of first dataset.")
with config.copy():
    cfg.build_env()

print("Forcing creation of second dataset.")
with config.copy(do_train=False, n_train=16, min_chars=1, postprocessing="", preserve_env=False):
    cfg.build_env()

run_kwargs = dict(
    n_repeats=1,
    kind="slurm",
    pmem=5000,
    ignore_gpu=False,
)

parser = argparse.ArgumentParser()
parser.add_argument("kind", choices="long_cedar long_graham short_graham short_cedar other".split())
args, _ = parser.parse_known_args()
kind = args.kind

if kind == "long_cedar":
    kind_args = dict(
        max_hosts=2, ppn=12, cpp=1, gpu_set="0,1,2,3", wall_time="24hours", project="rrg-dprecup",
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

elif kind == "other":
    kind_args = dict(
        max_hosts=1, ppn=3, cpp=2, gpu_set="0", wall_time="12hours", project="def-jpineau",
        cleanup_time="30mins", slack_time="30mins", n_param_settings=3)

else:
    raise Exception("Unknown kind: {}".format(kind))

run_kwargs.update(kind_args)

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(
    name="grid_task_param_search_{}".format(kind), config=config,
    distributions=distributions, **run_kwargs)
