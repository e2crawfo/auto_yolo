from auto_yolo import envs
from yolo_air_stage2 import durations, distributions, get_config

readme = "Running second stage (math learning) for ground_truth on addition task."

config = get_config("", "")

envs.run_experiment(
    "addition-stage2", config, readme, alg="ground_truth_math",
    task="arithmetic2", durations=durations, distributions=distributions
)
