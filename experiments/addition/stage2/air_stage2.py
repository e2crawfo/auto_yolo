from auto_yolo import envs
from yolo_air_stage2 import durations, distributions, get_config

readme = "Running second stage (math learning) for air on addition task."
config = get_config("", "")
config.update(
    max_time_steps=5, run_all_time_steps=True,
    scale_prior_mean=-2.944, scale_prior_std=0.75, shift_prior_std=0.5)

envs.run_experiment(
    "addition-stage2", config, readme, alg="air_math",
    task="arithmetic2", durations=durations, distributions=distributions
)
