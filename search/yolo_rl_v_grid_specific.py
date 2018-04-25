import clify
import argparse

from dps import cfg
from dps.config import DEFAULT_CONFIG

from dps.projects.nips_2018.envs import grid_config as env_config
from dps.projects.nips_2018.algs import yolo_rl_config as alg_config


def prepare_func():
    from dps import cfg

    cfg.area_neighbourhood_size = cfg.neighbourhood_size
    cfg.nonzero_neighbourhood_size = cfg.neighbourhood_size
    cfg.kernel_size = (cfg.kernel_size, cfg.kernel_size)


distributions = dict(
    neighbourhood_size=[0, 1],
    local_reconstruction_cost=[True, False],
    kernel_size=[1, 3],
)
distributions['sequential_cfg:on'] = [True, False]

config = DEFAULT_CONFIG.copy()

config.update(alg_config)
config.update(env_config)
config.curriculum[-1]['max_experiences'] = 100000000

config.update(
    prepare_func=prepare_func,
    render_step=100000,
    eval_step=1000,
    per_process_gpu_memory_fraction=0.2,

    max_experiences=100000,
    patience=10000000,
    max_steps=1000000,
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
        max_hosts=1, ppn=16, cpp=1, gpu_set="0,1,2,3", wall_time="6hours",
        cleanup_time="30mins", slack_time="30mins", n_param_settings=16,)

elif kind == "long_graham":
    kind_args = dict(
        max_hosts=2, ppn=8, cpp=2, gpu_set="0,1", wall_time="6hours", project="def-jpineau",
        cleanup_time="30mins", slack_time="30mins", n_param_settings=16)

elif kind == "short_cedar":
    kind_args = dict(
        max_hosts=1, ppn=4, cpp=1, gpu_set="0", wall_time="20mins",
        cleanup_time="2mins", slack_time="2mins", n_param_settings=4)

elif kind == "short_graham":
    kind_args = dict(
        max_hosts=1, ppn=4, cpp=1, gpu_set="0", wall_time="20mins", project="def-jpineau",
        cleanup_time="2mins", slack_time="2mins", n_param_settings=4)

else:
    raise Exception("Unknown kind: {}".format(kind))

run_kwargs.update(kind_args)

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(
    name="grid_task_specific_{}".format(kind), config=config,
    distributions=distributions, **run_kwargs)
