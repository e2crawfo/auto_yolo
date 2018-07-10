from auto_yolo import envs

readme = "Running DAIR experiment."

distributions = [
    {'n_digits': 1,
     'rnn_n_units': 128,
     'scale_prior_mean': -2.9444389791664403,
     'scale_prior_std': 0.4,
     'shift_prior_std': 1.0},

    {'n_digits': 3,
     'rnn_n_units': 64,
     'scale_prior_mean': -2.197224577336219,
     'scale_prior_std': 0.2,
     'shift_prior_std': 4.0},

    {'n_digits': 5,
     'rnn_n_units': 128,
     'scale_prior_mean': -2.197224577336219,
     'scale_prior_std': 0.2,
     'shift_prior_std': 4.0},

    {'n_digits': 7,
     'rnn_n_units': 64,
     'scale_prior_mean': -1.3862943611198906,
     'scale_prior_std': 0.8,
     'shift_prior_std': 4.0},

    {'n_digits': 9,
     'rnn_n_units': 128,
     'scale_prior_mean': -1.3862943611198906,
     'scale_prior_std': 1.0,
     'shift_prior_std': 0.5}
]


for dist in distributions:
    n_digits = dist['n_digits']
    dist.update(
        min_digits=n_digits,
        max_digits=n_digits,
        max_time_steps=n_digits)


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
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=1, n_param_settings=4),
)

config = dict(
    curriculum=[dict()],
    n_train=64000, run_all_time_steps=True,
    stopping_criteria="AP,max", threshold=0.99, patience=50000,
)


envs.run_experiment(
    "dair_run", config, readme, distributions=distributions,
    alg="dair", task="arithmetic", durations=durations,
)
