from auto_yolo import envs
from yolo_air_stage1 import durations, distributions, config

readme = "Running first stage (representation learning) for air on addition task."
config.update(
    max_time_steps=5, run_all_time_steps=True,
    scale_prior_mean=-2.944, scale_prior_std=0.75, shift_prior_std=0.5)

envs.run_experiment(
    "addition-stage1", config, readme, alg="air",
    task="arithmetic2", durations=durations, distributions=distributions
)
