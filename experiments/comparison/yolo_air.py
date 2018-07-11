import numpy as np
from auto_yolo import envs

readme = "testing yolo_air on scatter task for increasing numbers of digits"

durations = dict()

n_digits = 9


config = dict(
    curriculum=[dict()],
    n_train=64000, min_digits=n_digits, max_digits=n_digits, stopping_criteria="AP,max", threshold=1.0,
    count_prior_dist=0.01/17 * np.ones(17) + .99 * np.eye(17)[9, :]
)


envs.run_experiment(
    "air_arithmetic", config,
    readme, alg="yolo_air", task="arithmetic", durations=durations,
)
