from dps.utils.tf import MLP
from auto_yolo import envs

readme = "Running simple on CLEVR task."

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

config = dict(
    n_train=1000,
    clevr_background_mode=None,
    postprocessing="random",
    tile_shape=(48, 48),
    A=1,
    attr_prior_std=10.0,
    build_encoder=lambda scope: MLP(n_units=[100, 100], scope=scope),
    build_decoder=lambda scope: MLP(n_units=[100, 100], scope=scope),
)

envs.run_experiment(
    "clevr", config, readme, alg="simple",
    task="clevr", durations=durations, distributions=distributions
)
