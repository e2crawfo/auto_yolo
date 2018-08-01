from auto_yolo import envs

readme = "Testing yolo_air."

distributions = None

durations = dict(
    long=dict(
        max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", wall_time="24hours",
        project="rpp-bengioy", cleanup_time="20mins",
        slack_time="5mins", n_repeats=6, step_time_limit="24hours"),

    build=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="2hours",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, step_time_limit="2hours",
        config=dict(do_train=False)),

    short=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=1, n_param_settings=4),
)

config = dict(
    # training_wheels="Exp(1.0, 0.0, decay_rate=0.0, decay_steps=100, staircase=True)",
    n_train=16000, # obj_temp=100.0,
    # obj_logit_scale=1.0,
    # alpha_logit_scale=1.0,
    # alpha_logit_bias=1.0,
    # use_concrete_kl=True,
    # final_count_prior_log_odds=1000.0,
)

envs.run_experiment(
    "test", config, readme, alg="yolo_air",
    task="small", durations=durations, distributions=distributions,
)
