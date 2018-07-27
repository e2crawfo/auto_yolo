from auto_yolo import envs
from yolo_air_stage2 import durations, distributions, get_config

readme = "Running second stage (math learning) for baseline on addition task."

config = get_config("", "")
config.update(cc_threshold=0.02)  # Optimal for 5 digits.

envs.run_experiment(
    "addition-stage2", config, readme, alg="baseline_math",
    task="arithmetic2", durations=durations, distributions=distributions
)
