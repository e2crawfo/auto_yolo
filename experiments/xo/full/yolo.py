import time

from dps.utils import Config
from dps.utils.tf import MLP

from auto_yolo import envs
from auto_yolo.models import yolo_math

readme = "xo 2stage experiment"

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


config = config.copy(
    build_math_network=lambda scope: MLP([100, 100, 100], scope=scope),
)


envs.run_experiment(
    "yolo_xo_2stage_mlp", config, readme, alg="yolo_xo_2stage_mlp", task="xo",
    duration=durations,
)

# time.sleep(1)
# 
# config = config.copy(
#     build_math_network=yolo_math.SequentialRegressionNetwork,
# )
# 
# envs.run_experiment(
#     "yolo_xo_2stage_seq", config, readme, alg="yolo_xo_2stage_seq", task="xo",
#     duration=durations,
# )
# 
# time.sleep(1)
# 
# config = config.copy(
#     build_math_network=yolo_math.AttentionRegressionNetwork,
# )
# 
# envs.run_experiment(
#     "yolo_xo_2stage_attn", config, readme, alg="yolo_xo_2stage_attn", task="xo",
#     durations=durations,
# )
