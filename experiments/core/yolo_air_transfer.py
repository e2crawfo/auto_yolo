import clify
import argparse

from dps.config import DEFAULT_CONFIG

from dps.projects.nips_2018 import envs
from dps.projects.nips_2018.algs import yolo_air_config


parser = argparse.ArgumentParser()
parser.add_argument("duration", choices="long med".split())
parser.add_argument("size", choices="14 21".split())
parser.add_argument("--c", action="store_true")
args, _ = parser.parse_known_args()
duration = args.duration


distributions = [
    dict(min_chars=1, max_chars=5),
    dict(min_chars=6, max_chars=10),
    dict(min_chars=11, max_chars=15),
]


env_config = envs.get_mnist_config(size=args.size, colour=args.c, task="scatter")


config = DEFAULT_CONFIG.copy()
config.update(yolo_air_config)
config.update(env_config)
config.update(
    render_step=1000,
    eval_step=1000,
    per_process_gpu_memory_fraction=0.3,

    patience=1000000,
    max_experiences=100000000,
    max_steps=110000,

    count_prior_decay_steps=1000,
    final_count_prior_log_odds=0.0125,
    hw_prior_std=0.5,
    kernel_size=1,

    curriculum=[
        dict(),
    ] + [dict(preserve_env=False, do_train=False, n_train=32, n_val=200, min_chars=i, max_chars=i) for i in range(1, 16)]
)

config.log_name = "transfer_experiment_{}_alg={}".format(env_config.log_name, yolo_air_config.log_name)

run_kwargs = dict(
    n_repeats=8,
    kind="slurm",
    pmem=5000,
    ignore_gpu=False,
)

if duration == "long":
    duration_args = dict(
        max_hosts=2, ppn=12, cpp=2, gpu_set="0,1,2,3", wall_time="7hours", project="rpp-bengioy",
        cleanup_time="15mins", slack_time="15mins")

elif duration == "med":
    config.max_steps=1000
    duration_args = dict(
        max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", wall_time="1hour", project="rpp-bengioy",
        cleanup_time="10mins", slack_time="10mins", n_param_settings=3, n_repeats=2)
else:
    raise Exception("Unknown duration: {}".format(duration))

run_kwargs.update(duration_args)

readme = "Running the scatter transfer learning experiment with yolo_air."

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(
    name=config.log_name + "_duration={}".format(duration),
    config=config, readme=readme, distributions=distributions, **run_kwargs)
