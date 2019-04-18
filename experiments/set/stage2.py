import argparse

from auto_yolo import envs
from auto_yolo import algs
from dps.utils.tf import IdentityFunction
from dps.utils import Config

distributions = dict(decoder_kind="mlp recurrent obj arn1 arn3".split())

durations = dict(
    long=dict(
        max_hosts=1, ppn=12, cpp=2, gpu_set="0,1,2,3", wall_time="12hours",
        project="rpp-bengioy", cleanup_time="20mins",
        slack_time="5mins", n_repeats=7, step_time_limit="12hours"),

    build=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="2hours",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, step_time_limit="2hours",
        config=dict(do_train=False)),

    short=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="1mins", config=dict(max_steps=100),
        slack_time="1mins", n_repeats=1, n_param_settings=4),

    oak=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="1mins", config=dict(max_steps=100),
        slack_time="1mins", n_repeats=1, n_param_settings=5,
        kind="parallel", host_pool=[":"]),
)


def prepare_func():
    from dps import cfg
    from auto_yolo.models import networks
    from dps.utils.tf import AttentionalRelationNetwork, ObjectNetwork, MLP
    import tensorflow as tf

    decoder_kind = cfg.decoder_kind
    if decoder_kind == "mlp":
        cfg.build_math_network = lambda scope: MLP(n_units=[256, 256, 256, 256, 256], scope=scope)
        cfg.max_possible_objects = 7
    elif decoder_kind == "recurrent":
        cfg.build_math_network = networks.SimpleRecurrentRegressionNetwork
        cfg.build_math_cell = lambda scope: tf.contrib.rnn.LSTMBlockCell(cfg.n_recurrent_units)
    elif decoder_kind == "obj":
        cfg.build_math_network = ObjectNetwork
        cfg.build_on_input_network = lambda scope: MLP(n_units=[256, 256], scope=scope)
        cfg.build_on_object_network = lambda scope: MLP(n_units=[256, 256, 256], scope=scope)
        cfg.build_on_output_network = lambda scope: MLP(n_units=[256, 256, 256], scope=scope)
    elif decoder_kind == "arn1":
        cfg.n_repeats = 1
    elif decoder_kind == "arn3":
        cfg.n_repeats = 3
    else:
        raise Exception("Unknown value for decoder_kind: '{}'".format(decoder_kind))

    if decoder_kind.startswith("arn"):
        cfg.build_math_network = AttentionalRelationNetwork
        cfg.build_arn_network = lambda scope: MLP(n_units=[256, 256], scope=scope)
        cfg.build_arn_object_network = lambda scope: MLP(n_units=[256, 256, 256], scope=scope)

    if "stage1_path" in cfg:
        import os, glob  # noqa
        pattern = os.path.join(
            cfg.stage1_path, "experiments",
            "*idx={}*repeat={}*".format(cfg.idx, cfg.repeat))
        paths = glob.glob(pattern)
        assert len(paths) == 1
        cfg.load_path = {"network/representation": os.path.join(paths[0], "weights", "best_of_stage_0")}


config = Config(
    max_steps=2e5, patience=20000, idx=0, repeat=0,
    prepare_func=prepare_func,
    math_input_network=IdentityFunction,

    # recurrent
    n_recurrent_units=256,

    # obj / arn
    n_repeats=1,
    d=256,
    symmetric_op="max",
    layer_norm=True,
    use_mask=True,

    decoder_kind="mlp",
)


stage1_paths = dict(
    yolo_air="/media/data/dps_data/logs/set_env=task=set/exp_alg=yolo-air_seed=1947474600_2018_08_29_19_37_34"
)


extra_config = dict(
    yolo_air=dict(prepare_func=[prepare_func, algs.yolo_air_config.prepare_func]),
)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-kind", choices="raw fixed unfixed".split())
    parser.add_argument("--alg", choices="yolo_air air simple ground_truth baseline".split())
    args, _ = parser.parse_known_args()
    alg = args.alg + "_math"

    _extra_config = extra_config.get(args.alg, {})
    config.update(_extra_config)

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

    readme = "Running second stage for {} on set task.".format(alg)
    envs.run_experiment(
        "set-stage2", config, readme, alg=alg, task="set",
        durations=durations, distributions=distributions, name_variables="run_kind",
    )


run()
