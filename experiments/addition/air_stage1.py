from auto_yolo import envs

readme = "Running first stage (representation learning) for air on addition task."

distributions = [dict(n_train=1000 * 2**i) for i in range(8)]


durations = dict(
    long=dict(
        max_hosts=1, ppn=12, cpp=2, gpu_set="0,1,2,3", project="rpp-bengioy",
        wall_time="48hours", cleanup_time="5mins", slack_time="5mins",
        n_repeats=6, step_time_limit="48hours"),

    short=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=1, n_param_settings=4),
)

config = dict(
    n_train=16000, max_time_steps=5, run_all_time_steps=True,
    scale_prior_mean=-2.944, scale_prior_std=0.75, shift_prior_std=0.5)

envs.run_experiment(
    "addition-stage1", config, readme, alg="air",
    task="arithmetic2", durations=durations, distributions=distributions
)
