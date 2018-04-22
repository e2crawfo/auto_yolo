import clify
import argparse

from dps import cfg
from dps.config import DEFAULT_CONFIG

from dps.projects.nips_2018.envs import grid_fullsize_config as env_config
from dps.projects.nips_2018.algs import air_config as alg_config

distributions = dict()

config = DEFAULT_CONFIG.copy()

config.update(alg_config)
config.update(env_config)

config.update(
    render_step=100000,
    eval_step=1000,

    max_experiences=10000000,
    patience=10000000,
    max_steps=1000000,

    # Train directly on the large images.
    postprocessing="",
)

config.log_name = "{}_VERSUS_{}".format(alg_config.log_name, env_config.log_name)

print("Forcing creation of first dataset.")
with config.copy(mock_load=True):
    cfg.build_env()

print("Forcing creation of second dataset.")
with config.copy(config.curriculum[-1], mock_load=True):
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
        max_hosts=2, ppn=8, cpp=1, gpu_set="0,1", wall_time="6hours", project="def-jpineau",
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
    name="grid_task_param_search_{}".format(kind), config=config,
    distributions=distributions, **run_kwargs)
