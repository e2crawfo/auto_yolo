from auto_yolo import envs
import argparse

readme = "Running second stage (math learning) for yolo_air on addition task."

distributions = [dict(n_train=1000 * 2**i) for i in range(8)]

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


def get_config(fixed_path, unfixed_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices="raw fixed unfixed".split())
    args, _ = parser.parse_known_args()

    config = dict(max_steps=2e5, patience=10000, n_train=16000)

    if args.kind == "raw":
        pass
    elif args.kind == "fixed":
        config.update(
            fixed_weights="object_encoder object_decoder box obj backbone",
            # load_directory="",
        )
    elif args.kind == "unfixed":
        config.update(
            fixed_weights="object_decoder box obj backbone",
            # load_directory="",
        )
    else:
        raise Exception("Unknown kind: {}".format(args.kind))

    return config


if __name__ == "__main__":
    config = get_config("", "")
    envs.run_experiment(
        "addition-stage2", config, readme, alg="yolo_air_math",
        task="arithmetic2", durations=durations, distributions=distributions
    )
