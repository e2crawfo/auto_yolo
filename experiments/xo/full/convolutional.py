from dps.utils import Config

from auto_yolo import envs

readme = "xo convolutional experiment"

config = Config()

durations = dict(
    long=dict(
        max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", wall_time="4hours",
        project="rpp-bengioy", cleanup_time="10mins",
        slack_time="10mins", n_repeats=6),
    med=dict(
        max_hosts=1, ppn=3, cpp=2, gpu_set="0", wall_time="30mins",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=3),
    short=dict(
        max_hosts=1, ppn=3, cpp=2, gpu_set="0", wall_time="10mins",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=3)
)

envs.run_experiment(
    "conv_xo", config, readme, alg="yolo_xo_simple", task="xo",
    name_variables="decoder_kind", durations=durations,
)
