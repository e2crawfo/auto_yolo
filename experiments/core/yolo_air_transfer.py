from auto_yolo import envs

readme = "redoing yolo_air transfer experiment"

durations = dict(
    long=dict(
        max_hosts=1, ppn=4, cpp=2, gpu_set="0,1", wall_time="24hours",
        project="rpp-bengioy", cleanup_time="20mins",
        slack_time="5mins", n_repeats=8, step_time_limit="24hours"),
    build=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="2hours",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, step_time_limit="2hours"),
    short=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="4hours",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, step_time_limit="4hours"),
)

envs.run_experiment(
    "yolo_air_transfer", dict(), readme,
    alg="yolo_air_transfer", task="scatter", durations=durations,
)
