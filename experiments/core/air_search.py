import numpy as np

from auto_yolo import envs

readme = "searching for best hyperparameters for varying numbers of digits"

p = np.array([0.05, 0.1, 0.2, 0.4])
distributions = dict(
    scale_prior_mean=np.log(p/(1-p)),
    scale_prior_std=[.2, .4, .6, .8, 1.0],
    shift_prior_std=[0.5, 1.0, 2.0, 4.0],
    rnn_n_units=[64, 128, 256],
)

durations = dict(
    long=dict(
        max_hosts=1, ppn=12, cpp=2, gpu_set="0,1", wall_time="24hours",
        project="rpp-bengioy", cleanup_time="20mins",
        slack_time="5mins", n_repeats=1, step_time_limit="24hours"),

    build=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, step_time_limit="2hours",
        config=dict(do_train=False)),

    short=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=1, n_param_settings=4),

    small_oak=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="30mins",
        cleanup_time="1mins", n_param_settings=4,
        slack_time="1mins", n_repeats=1, kind="parallel", host_pool=":"),

    build_oak=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="30mins",
        cleanup_time="1mins", n_param_settings=1,
        slack_time="1mins", n_repeats=1, kind="parallel", host_pool=":",
        config=dict(do_train=False)),
)

n_digits = 1

config = dict(
    curriculum=[dict()],
    n_train=64000, min_digits=n_digits, max_digits=n_digits,
    max_time_steps=n_digits, run_all_time_steps=True,
    stopping_criteria="AP_at_point_25,max",
    threshold=0.99,
)


envs.run_experiment(
    "air_search_n_digits={}".format(n_digits), config, readme,
    distributions=distributions, alg="air", task="arithmetic", durations=durations,
)
