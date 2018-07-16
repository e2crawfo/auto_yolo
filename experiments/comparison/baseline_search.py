from auto_yolo import envs
from auto_yolo.models import yolo_air
import argparse
import numpy as np

readme = "Searching for baseline threshold."


parser = argparse.ArgumentParser()
parser.add_argument("--n-digits", type=int, default=1)
args, _ = parser.parse_known_args()

dist_dict = {
    3: np.linspace(0, .1, 101),
    5: np.linspace(0, .1, 101),
    7: np.linspace(.6599-0.05, .6599+0.05, 101),
    9: np.linspace(.599-0.05, .599+0.05, 101),
}

distributions = [dict(cc_threshold=t) for t in dist_dict[args.n_digits]]

durations = dict(
    oak=dict(
        max_hosts=1, ppn=4, cpp=1, gpu_set="0", wall_time="1year",
        cleanup_time="1mins", slack_time="1mins", n_repeats=1, kind="parallel", host_pool=":"),
)

config = dict(
    curriculum=[dict()],
    n_train=64000, run_all_time_steps=True,
    stopping_criteria="AP,max", threshold=0.99, patience=50000,
    n_digits=args.n_digits,
    min_digits=args.n_digits,
    max_digits=args.n_digits,
    render_hook=None,
    cc_threshold=1e-3
)

envs.run_experiment(
    "baseline_search", config, readme, distributions=distributions,
    alg="baseline", task="arithmetic", durations=durations,
)
