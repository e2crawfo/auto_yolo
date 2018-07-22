from auto_yolo import envs

readme = "Testing air variational autoencoder with math."

distributions = None

durations = dict()

n_digits = 2
largest_digit = n_digits * 9
n_classes = largest_digit + 1

config = dict(
    n_train=16000, min_digits=n_digits, max_digits=n_digits,
    max_time_steps=n_digits, run_all_time_steps=True,
    largest_digit=largest_digit, n_classes=n_classes)

envs.run_experiment(
    "test_math", config, readme, alg="air_2stage_math",
    task="arithmetic", durations=durations, distributions=distributions
)
