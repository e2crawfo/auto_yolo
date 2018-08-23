import numpy as np
from auto_yolo import envs

readme = "Parameter search for yolo_air on clevr."

count_prior_probs = np.array([0.25, 0.167, 0.111, 0.074])
hw_prior_probs = np.array([0.4, 0.3, 0.2, 0.1])

distributions = dict(
    final_count_prior_log_odds=np.log(count_prior_probs / (1-count_prior_probs)),
    hw_prior_mean=np.log(hw_prior_probs / (1-hw_prior_probs)),
    hw_prior_std=[0.25, 0.5, 0.75, 1.0],
    alpha_logit_scale=[1.0, 0.5, 0.25, 0.125],
    count_prior_decay_steps=[1000, 2000, 3000],
)

durations = dict(
    long=dict(
        max_hosts=2, ppn=16, cpp=1, gpu_set="0,1,2,3", wall_time="48hours",
        cleanup_time="5mins", slack_time="5mins", project="rpp-bengioy",
        n_repeats=1, n_param_settings=192),

    build=dict(
        max_hosts=1, ppn=1, cpp=4, gpu_set="0", wall_time="2hours",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, step_time_limit="2hours",
        config=dict(do_train=False), n_param_settings=1),

    short=dict(
        max_hosts=1, ppn=4, cpp=1, gpu_set="0", wall_time="40mins",
        project="rpp-bengioy", cleanup_time="2mins", config=dict(max_steps=2000),
        slack_time="2mins", n_repeats=1, n_param_settings=8),

    oak=dict(
        host_pool=[":"], kind="parallel",
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="1hour",
        project="rpp-bengioy", cleanup_time="1mins", slack_time="1mins",
        step_time_limit="1hour", n_repeats=10, n_param_settings=1,
        config=dict(max_steps=4000)),
)

config = dict(
    background_cfg=dict(mode="learn", A=1),
    obj_logit_scale=1.0,
    alpha_logit_scale=1.0,
    alpha_logit_bias=1.0,
    obj_temp=1.0,
    # training_wheels=0.0,
    hw_prior_mean=np.log(.33 / .67),
    hw_prior_std=1.0,
    max_steps=10000000,
    patience=10000000,
    final_count_prior_log_odds=0.1,
    postprocessing="random",

    n_train=70000,
    clevr_background_mode=None,
    tile_shape=(48, 48),
    image_shape=(80, 120),
    pixels_per_cell=(12, 12),
    object_shape=(28, 28),

    val_example_range=(0, 2500),
    test_example_range=(2500, 5000),
)

envs.run_experiment(
    "yolo_air_clevr_search", config, readme, alg="yolo_air",
    task="clevr", durations=durations, distributions=distributions,
)
