from auto_yolo import envs

readme = "Searching for YOLO_AIR parameters."

durations = dict(
    long=dict(
        max_hosts=1, ppn=16, cpp=1, gpu_set="0,1,2,3", wall_time="27hours",
        project="rpp-bengioy", cleanup_time="10mins",
        slack_time="10mins", n_repeats=3, n_param_settings=3*16),

    build=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, step_time_limit="20mins", n_param_settings=1,
        config=dict(do_train=False)),

    short=dict(
        max_hosts=1, ppn=3, cpp=2, gpu_set="0", wall_time="30mins",
        project="rpp-bengioy", cleanup_time="2mins", n_param_settings=3,
        slack_time="2mins", n_repeats=1, config=dict(max_steps=100))
)

distributions = dict(
    kernel_size=[1, 2],
    hw_prior_std=[0.5, 1.0, 2.0],  # Anything outside of these bounds doesn't work very well.
    final_count_prior_log_odds=[0.1, 0.05, 0.025, 0.0125],
    count_prior_decay_steps=[1000, 2000, 3000, 4000],
)

config = dict(
    curriculum=[dict()],
    stopping_criteria="AP,max", threshold=0.99, patience=50000, max_steps=2e5,

    color_logit_scale=1.0,
    alpha_logit_scale=1.0,
    alpha_logit_bias=1.0,
    obj_temp=1.0,
    training_wheels=0.0,
)

envs.run_experiment(
    "yolo_air_search_SET", config, readme, alg="yolo_air",
    task="set", durations=durations, distributions=distributions,
)
