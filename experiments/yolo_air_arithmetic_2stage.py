import clify
import argparse

from dps.config import DEFAULT_CONFIG

from dps.projects.nips_2018 import envs
from dps.projects.nips_2018.algs import yolo_math_config


parser = argparse.ArgumentParser()
parser.add_argument("duration", choices="long med supplement".split())
parser.add_argument("size", choices="14 21".split())
parser.add_argument("task", choices="addition arithmetic".split())
parser.add_argument("--c", action="store_true")
args, _ = parser.parse_known_args()
duration = args.duration


distributions = dict(
    n_train=[1000, 2000, 4000, 8000, 16000, 32000],
)

env_config = envs.get_mnist_config(size=args.size, colour=args.c, task=args.task)


config = DEFAULT_CONFIG.copy()
config.update(yolo_math_config)
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
    postprocessing="random",
    tile_shape=(48, 48),

    curriculum=[
        dict(math_weight=0.0, fixed_weights="math"),
        dict(
            max_steps=100000000,
            postprocessing="",
            preserve_env=False,
            math_weight=1.0,
            train_kl=False,
            train_reconstruction=False,
            fixed_weights="object_encoder object_decoder box obj backbone edge",
            stopping_criteria="math_accuracy,max",
            threshold=1.0,
        )
    ],
    robust=False,
)

config.log_name = "sample_complexity_experiment_{}_alg={}_2stage".format(env_config.log_name, yolo_math_config.log_name)

run_kwargs = dict(
    n_repeats=6,
    kind="slurm",
    pmem=5000,
    ignore_gpu=False,
)

readme = "Running a sample complexity experiment with yolo_air on the {} task.".format(args.task)


if duration == "long":
    duration_args = dict(
        max_hosts=3, ppn=12, cpp=2, gpu_set="0,1,2,3", wall_time="10hours", project="rpp-bengioy",
        cleanup_time="30mins", slack_time="30mins")

elif duration == "med":
    config.max_steps=1000
    duration_args = dict(
        max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", wall_time="1hour", project="rpp-bengioy",
        cleanup_time="10mins", slack_time="10mins", n_repeats=1)

elif duration == "supplement":
    distributions = dict(
        n_train=[2000],
    )
    config.log_name = "supplement_" + config.log_name
    duration_args = dict(
        max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", wall_time="10hours", project="rpp-bengioy",
        cleanup_time="30mins", slack_time="30mins")
    readme = "Supplemental learning experiment with yolo_air, gathering a data point that we missed previously."

else:
    raise Exception("Unknown duration: {}".format(duration))

run_kwargs.update(duration_args)

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(
    name=config.log_name + "_duration={}".format(duration),
    config=config, readme=readme, distributions=distributions, **run_kwargs)
