from auto_yolo import envs
from yolo_air_stage1 import durations, distributions, config

readme = "Running simple on addition task."

envs.run_experiment(
    "addition-stage1", config, readme, alg="simple",
    task="arithmetic2", durations=durations, distributions=distributions
)
