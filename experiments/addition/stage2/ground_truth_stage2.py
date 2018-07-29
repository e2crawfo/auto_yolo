from auto_yolo import envs
from yolo_air_stage2 import durations, distributions, get_config

readme = "Running second stage (math learning) for ground_truth on addition task."

stage1_path = "/scratch/e2crawfo/dps_data/run_experiments/run_search_addition-stage1_env=size=14-in-colour=False-task=arithmetic2_alg=ground-truth_duration=long_seed=0_2018_07_27_15_22_33"
config = get_config(stage1_path)

envs.run_experiment(
    "addition-stage2", config, readme, alg="ground_truth_math",
    task="arithmetic2", durations=durations, distributions=distributions,
    name_variables="run_kind",
)
