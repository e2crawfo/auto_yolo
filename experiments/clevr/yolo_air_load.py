import numpy as np
from dps.utils.tf import MLP
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

    oak=dict(
        host_pool=[":"], kind="parallel",
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="1hour",
        project="rpp-bengioy", cleanup_time="1mins", slack_time="1mins",
        step_time_limit="1hour", n_repeats=10, n_param_settings=1,
        config=dict(max_steps=4000)),
)

load_path = "/media/data/dps_data/logs/test-clevr_env=task=clevr/exp_alg=yolo-air_seed=181029782_2018_08_05_22_37_57/weights/final_for_stage_0"

config = dict(
    clevr_background_mode="mean",
    background_cfg=dict(mode="data"),
    # background_cfg=dict(mode="colour", colour="white"),
    obj_logit_scale=1.0,
    alpha_logit_scale=1.0,
    alpha_logit_bias=1.0,
    obj_temp=1.0,
    # training_wheels=0.0,
    tile_shape=(36, 36),
    hw_prior_mean=np.log(.25 / .75),
    patience=10000000,

    load_path={
        "network/representation/z_sequential_network": load_path,
        "network/representation/box_sequential_network": load_path,
        "network/representation/obj_sequential_network": load_path,
        "network/representation/backbone": load_path,
        "network/representation/object_encoder": load_path,
        "network/representation/object_decoder": load_path,
    },

    postprocessing="",
    max_steps=1,
    do_train=False,
    n_train=32,
    n_val=32,
)

envs.run_experiment(
    "test_clevr", config, readme, alg="yolo_air",
    task="clevr", durations=durations, distributions=distributions,
)
