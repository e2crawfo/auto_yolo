from auto_yolo import envs

readme = "Running ground_truth on addition task."

distributions = [dict(n_train=1000 * 2**i) for i in range(8)]

durations = dict(
    long=dict(
        max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", wall_time="24hours",
        project="rpp-bengioy", cleanup_time="5mins",
        slack_time="5mins", n_repeats=6, step_time_limit="24hours"),

    short=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=1, n_param_settings=4),
)

envs.run_experiment(
    "addition", dict(n_train=16000), readme, alg="ground_truth_math",
    task="arithmetic2", durations=durations, distributions=distributions
)
