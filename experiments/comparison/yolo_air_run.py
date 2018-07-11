from auto_yolo import envs

readme = "Running YOLO AIR experiment."

distributions = [
    dict(
        n_digits=n_digits,
        min_digits=n_digits,
        max_digits=n_digits,
        max_time_steps=n_digits)
    for n_digits in [1, 3, 5, 7, 9]]


durations = dict(
    long=dict(
        max_hosts=1, ppn=16, cpp=1, gpu_set="0,1,2,3", wall_time="24hours",
        project="rpp-bengioy", cleanup_time="10mins",
        slack_time="10mins", n_repeats=6, step_time_limit="24hours"),

    build=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, step_time_limit="2hours",
        config=dict(do_train=False), n_param_settings=1,),

    short=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, n_param_settings=4),
)

config = dict(
    curriculum=[dict()],
    n_train=64000, run_all_time_steps=True,
    stopping_criteria="AP,max", threshold=0.99, patience=50000,
)

envs.run_experiment(
    "yolo_air_run", config, readme, distributions=distributions,
    alg="yolo_air", task="arithmetic", durations=durations,
)
