from auto_yolo import envs

readme = "Testing air."

distributions = None

durations = dict()

config = dict(
    n_train=16000, min_digits=2, max_digits=2,
    max_time_steps=2, run_all_time_steps=True)

envs.run_experiment(
    "test", config, readme, alg="air",
    task="arithmetic", durations=durations, distributions=distributions
)
