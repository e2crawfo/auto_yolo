import clify
import argparse

from dps import cfg
from dps.config import DEFAULT_CONFIG

from dps.projects.nips_2018.envs import air_testing_config as env_config
# from dps.projects.nips_2018.envs import grid_fullsize_config as env_config
from dps.projects.nips_2018.algs import air_config as alg_config

distributions = [
    dict(cnn=True),
    dict(cnn=True, vae_likelihood_std=0.0),
    dict(),
    dict(vae_likelihood_std=0.0),
]

config = DEFAULT_CONFIG.copy()

config.update(alg_config)
config.update(env_config)

config.update(
    per_process_gpu_memory_fraction=0.45,
    render_step=5000,
    eval_step=1000,

    max_experiences=10000000,
    patience=10000000,
    max_steps=1000000,

    z_pres_prior_log_odds=10.0,
)

config.log_name = "{}_VERSUS_{}".format(alg_config.log_name, env_config.log_name)

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
parser.add_argument("kind", choices="long_cedar long_graham short_graham short_cedar other short_other".split())
args, _ = parser.parse_known_args()
kind = args.kind

if kind == "long_cedar":
    kind_args = dict(
        max_hosts=1, ppn=4, cpp=2, gpu_set="0,1", wall_time="24hours",
        cleanup_time="30mins", slack_time="30mins")

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

elif kind == "other":
    kind_args = dict(
        max_hosts=1, ppn=4, cpp=2, gpu_set="0,1", wall_time="6hours", project="def-jpineau",
        cleanup_time="30mins", slack_time="30mins", n_param_settings=4)

elif kind == "short_other":
    kind_args = dict(
        max_hosts=1, ppn=4, cpp=2, gpu_set="0,1", wall_time="30mins", project="def-jpineau",
        cleanup_time="3mins", slack_time="3mins", n_param_settings=4)

else:
    raise Exception("Unknown kind: {}".format(kind))

run_kwargs.update(kind_args)

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(
    name="AIR_v_grid_task_param_search_{}".format(kind), config=config,
    distributions=distributions, **run_kwargs)
