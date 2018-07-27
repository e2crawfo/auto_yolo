import numpy as np
import argparse

from auto_yolo import envs

readme = "searching for best hyperparameters for varying numbers of digits"

p = np.array([0.05, 0.1, 0.2, 0.4])
distributions = dict(
    scale_prior_mean=np.log(p/(1-p)),
    scale_prior_std=[.25, .5, .75, 1.0],
    shift_prior_std=[0.5, 1.0, 2.0],
)

parser = argparse.ArgumentParser()
parser.add_argument("--n-digits", type=int, default=1)
parser.add_argument("--dair", action="store_true")
args, _ = parser.parse_known_args()


durations = dict(
    long=dict(
        max_hosts=1, ppn=16, cpp=1, gpu_set="0,1,2,3", wall_time="48hours",
        project="rpp-bengioy", cleanup_time="5mins",
        slack_time="5mins", n_repeats=3, step_time_limit="48hours"),

    build=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, step_time_limit="2hours",
        config=dict(do_train=False), n_param_settings=1,),

    short=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="1mins", config=dict(max_steps=100),
        slack_time="1mins", n_repeats=1, n_param_settings=4),

    oak=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="12mins",
        project="rpp-bengioy", cleanup_time="1mins", config=dict(max_steps=100),
        slack_time="1mins", n_repeats=1, n_param_settings=4, host_pool=[":"],
        kind="parallel"
    ),
)

n_digits = args.n_digits

config = dict(
    curriculum=[dict()],
    n_train=64000, min_digits=n_digits, max_digits=n_digits,
    max_time_steps=n_digits, run_all_time_steps=True, patience=10000,
    train_example_range=(0.0, 0.7),
    val_example_range=(0.7, 0.8),
    test_example_range=(0.8, 0.9),
)

alg = "dair" if args.dair else "air"

envs.run_experiment(
    "{}_search_n_digits={}".format(alg, n_digits), config, readme,
    distributions=distributions, alg=alg, task="arithmetic", durations=durations,
)
