from auto_yolo import envs
from yolo_air_stage2 import durations, distributions, get_config

readme = "Running second stage (math learning) for air on addition task."
stage1_path = "/scratch/e2crawfo/dps_data/run_experiments/run_search_addition-stage1_env=size=14-in-colour=False-task=arithmetic2_alg=attend-infer-repeat_duration=long_seed=0_2018_07_27_15_24_22"
config = get_config(stage1_path)
config.update(
    max_time_steps=5, run_all_time_steps=True,
    scale_prior_mean=-2.944, scale_prior_std=0.75, shift_prior_std=0.5)

envs.run_experiment(
    "addition-stage2", config, readme, alg="air_math",
    task="arithmetic2", durations=durations, distributions=distributions,
    name_variables="run_kind",
)
