from auto_yolo import envs
import argparse

readme = "Second stage of testing yolo_air variational autoencoder with math."

distributions = None

durations = dict(
    long=dict(
        max_hosts=1, ppn=12, cpp=2, gpu_set="0,1,2,3", wall_time="24hours",
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

parser = argparse.ArgumentParser()
parser.add_argument("--fixed", action="store_true")
args, _ = parser.parse_known_args()

if args.fixed:
    fixed_weights = "object_encoder object_decoder box obj backbone"
else:
    fixed_weights = "object_decoder box obj backbone"

config = dict(
    n_train=16000,
    load_path={
        "network/representation":
            "/media/data/dps_data/logs/test-yolo-air-math_env=size=14-in-colour=False-task=arithmetic-ops=addition/"
            "exp_alg=yolo-air-math_seed=174419635_2018_07_18_16_20_00/weights/best_of_stage_0",
    },
    curriculum=[dict()],
    fixed_weights=fixed_weights
)

envs.run_experiment(
    "test_math", config, readme, alg="yolo_air_2stage_math",
    task="arithmetic2", durations=durations, distributions=distributions
)
