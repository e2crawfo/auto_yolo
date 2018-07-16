import numpy as np
import argparse

from auto_yolo import envs

readme = "searching for best hyperparameters for varying numbers of digits"

p = np.array([0.05, 0.1, .2, .4])
distributions = dict(
    scale_prior_mean=np.log(p/(1-p)),
    complete_rnn_input=[True, False],
)

parser = argparse.ArgumentParser()
parser.add_argument("--n-digits", type=int, default=1)
args, _ = parser.parse_known_args()


durations = dict(
    oak=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="5mins", step_time_limit="5mins",
        cleanup_time="1mins", slack_time="1mins", n_repeats=1, kind="parallel", host_pool=":")
)

n_digits = args.n_digits

config = dict(
    curriculum=[dict()],
    n_train=1000, min_digits=n_digits, max_digits=n_digits,
    max_time_steps=n_digits, run_all_time_steps=True,
    stopping_criteria="AP,max", threshold=0.99, patience=10000,
    rnn_n_units=128,
)

envs.run_experiment(
    "air_search_n_digits={}".format(n_digits), config, readme,
    distributions=distributions, alg="air", task="arithmetic", durations=durations,
)
