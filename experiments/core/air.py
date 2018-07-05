import numpy as np

from auto_yolo import envs

readme = "testing AIR on scatter task for increasing numbers of digits"

durations = dict()

n_digits = 1


config = dict(
    # scale_prior_mean=np.log(14/24),
    # scale_prior_mean=np.log(.1/.9),
    # scale_prior_std=0.5,
    z_pres_prior_log_odds=np.log(10000.0),
    curriculum=[dict()],
    n_train=64000, min_digits=n_digits, max_digits=n_digits,
    max_time_steps=n_digits, verbose_summaries=False,
)


envs.run_experiment(
    "air_arithmetic", config,
    readme, alg="air", task="arithmetic", durations=durations,
)
