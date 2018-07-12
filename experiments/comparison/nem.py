import numpy as np

from auto_yolo import envs

readme = "testing NEM on scatter task for increasing numbers of digits"

durations = dict()

n_digits = 3


config = dict(
    z_pres_prior_log_odds=np.log(10000.0),
    curriculum=[dict()],
    n_train=64000, min_digits=n_digits, max_digits=n_digits,
    max_time_steps=n_digits, run_all_time_steps=True,
)


envs.run_experiment(
    "air_arithmetic", config,
    readme, alg="nem", task="arithmetic", durations=durations,
)
