from auto_yolo import envs
from yolo_air_stage1 import durations, distributions, config

readme = "Running first stage (representation learning) for baseline on addition task."

config.update(cc_threshold=0.02)  # Optimal for 5 digits.

envs.run_experiment(
    "addition-stage1", config, readme, alg="baseline",
    task="arithmetic2", durations=durations, distributions=distributions
)
