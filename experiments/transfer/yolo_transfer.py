from auto_yolo import envs

readme = "redoing yolo_air transfer experiment"

distributions = [
    dict(min_chars=1, max_chars=5),
    dict(min_chars=6, max_chars=10),
    dict(min_chars=11, max_chars=15),
]

durations = dict(
    long=dict(
        max_hosts=1, ppn=12, cpp=2, gpu_set="0,1,2,3", wall_time="48hours",
        project="rpp-bengioy", cleanup_time="20mins",
        slack_time="5mins", n_repeats=8, step_time_limit="48hours"),

    build=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="3hours",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, step_time_limit="3hours", n_param_settings=1,
        config=dict(
            do_train=False,
            curriculum=[
                dict(min_chars=1, max_chars=5, postprocessing="random"),
                dict(min_chars=6, max_chars=10, postprocessing="random"),
                dict(min_chars=11, max_chars=15, postprocessing="random")] + [
                dict(min_chars=n, max_chars=n, n_train=32, n_val=200, do_train=False) for n in range(1, 21)]),
    ),

    short=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=1, n_param_settings=4),

    small_oak=dict(
        max_hosts=1, ppn=4, cpp=2, gpu_set="0", wall_time="30mins",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=2, kind="parallel", host_pool=":"),

    build_oak=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="1year",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=1, kind="parallel", host_pool=":",
        config=dict(do_train=False)),

    oak=dict(
        max_hosts=1, ppn=4, cpp=2, gpu_set="0", wall_time="1year",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=6, kind="parallel", host_pool=":",
        step_time_limit="1year"),
)

envs.run_experiment(
    "yolo_air_transfer", dict(), readme,
    alg="yolo_air_transfer", task="scatter", durations=durations,
    distributions=distributions
)
