import numpy as np

from dps.utils import Config

from auto_yolo import envs

readme = "xo pretrained representation experiment: training the decoders, assuming the representation has been trained"

config = Config()
distributions = dict(n_train=100 * 2 ** np.arange(8))

envs.run_experiment(
    "yolo_xo_continue", config, readme, alg="yolo_xo_continue", task="xo",
    distributions=distributions, name_variables="decoder_kind",
    durations=dict(
        long=dict(
            max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", wall_time="24hours",
            project="rpp-bengioy", cleanup_time="10mins",
            slack_time="5mins", n_repeats=6, step_time_limit="24hours"),
        med=dict(
            max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="1hour",
            project="rpp-bengioy", cleanup_time="5mins",
            slack_time="2mins", n_repeats=1, step_time_limit="1hour"),
        short=dict(
            max_hosts=1, ppn=3, cpp=2, gpu_set="0", wall_time="10mins",
            project="rpp-bengioy", cleanup_time="2mins",
            slack_time="2mins", n_repeats=3, n_param_settings=1)
    )
)
