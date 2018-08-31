from auto_yolo import envs
import argparse

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
        project="rpp-bengioy", cleanup_time="1mins", config=dict(max_steps=100),
        slack_time="1mins", n_repeats=1, n_param_settings=4),
)

stage1_paths = dict(
    yolo_air="/media/data/dps_data/logs/shapes-qa_env=task=shapes-qa/exp_alg=yolo-air_seed=1121983585_2018_08_28_13_14_11",
)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-kind", choices="raw fixed unfixed".split())
    parser.add_argument("--alg", choices="yolo_air air simple ground_truth baseline".split())
    args, _ = parser.parse_known_args()
    alg = args.alg + "_math"

    config = dict(max_steps=2e5, patience=10000, n_train=4000, idx=0, repeat=0)

    if args.run_kind == "raw":
        pass
    elif args.run_kind == "fixed":
        stage1_path = stage1_paths[args.alg]
        config.update(
            fixed_weights="encoder decoder object_encoder object_decoder box z obj backbone image_encoder cell output",
            stage1_path=stage1_path,
        )
    elif args.run_kind == "unfixed":
        stage1_path = stage1_paths[args.alg]
        config.update(
            fixed_weights="decoder object_decoder box z obj backbone image_encoder cell output",
            stage1_path=stage1_path,
        )
    else:
        raise Exception("Unknown kind: {}".format(args.kind))
    config['run_kind'] = args.run_kind

    readme = "Running second stage for {} on shapes qa task.".format(alg)
    envs.run_experiment(
        "shapes_qa-stage2", config, readme, alg=alg, task="shapes_qa",
        durations=durations, distributions=distributions, name_variables="run_kind",
    )


run()
