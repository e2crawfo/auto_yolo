from auto_yolo import envs
from auto_yolo import algs
import argparse
import copy

readme = "Running yolo_air on addition task."

distributions = [dict(n_train=1000 * 2**i) for i in range(8)]

durations = dict(
    long=dict(
        max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", wall_time="24hours",
        project="rpp-bengioy", cleanup_time="5mins",
        slack_time="5mins", n_repeats=6, step_time_limit="24hours"),

    short=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=1, n_param_settings=4),
)

parser = argparse.ArgumentParser()
parser.add_argument("--fixed", action="store_true")
args, _ = parser.parse_known_args()

if args.fixed:
    fixed_weights = "object_encoder object_decoder box obj backbone"
else:
    fixed_weights = "object_decoder box obj backbone"

config = dict(
    curriculum=copy.deepcopy(algs.yolo_air_2stage_math_config.curriculum),
)
config['curriculum'][1]['fixed_weights'] = fixed_weights

envs.run_experiment(
    "addition_fixed={}".format(args.fixed), config, readme, alg="yolo_air_2stage_math",
    task="arithmetic2", durations=durations, distributions=distributions
)
