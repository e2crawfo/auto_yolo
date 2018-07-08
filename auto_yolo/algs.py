import numpy as np
import tensorflow as tf
import copy

from dps import cfg
from dps.utils import Config
from dps.utils.tf import MLP, IdentityFunction, FullyConvolutional, FeedforwardCell

from auto_yolo.models import (
    core, air, yolo_air, yolo_math, yolo_xo, baseline
)


alg_config = Config(
    get_updater=core.Updater,

    batch_size=32,
    lr_schedule=1e-4,
    optimizer_spec="adam",
    max_grad_norm=1.0,
    use_gpu=True,
    gpu_allow_growth=True,
    eval_mode="val",
    max_experiments=None,
    preserve_env=False,
    stopping_criteria="loss,min",
    threshold=-np.inf,
    load_path=-1,

    max_steps=10000000,
    patience=50000,
    render_step=10000,
    eval_step=1000,
    display_step=1000,

    curriculum=[dict()],

    tile_shape=(48, 48),
    n_samples_per_image=4,
    postprocessing="",
)


class AirImageEncoder(FullyConvolutional):
    def __init__(self, **kwargs):
        layout = [
            dict(filters=128, kernel_size=3, strides=1, padding="VALID"),
            dict(filters=128, kernel_size=3, strides=1, padding="VALID"),
            dict(filters=128, kernel_size=3, strides=2, padding="SAME"),
            dict(filters=128, kernel_size=3, strides=2, padding="SAME"),
            dict(filters=16, kernel_size=3, strides=2, padding="SAME"),
        ]
        super(AirImageEncoder, self).__init__(layout, check_output_shape=False, **kwargs)


# Original config from tf-attend_infer_repeat
air_config = alg_config.copy(
    alg_name="attend_infer_repeat",
    build_network=air.AIR_Network,
    batch_size=64,
    verbose_summaries=False,
    render_hook=air.AIR_RenderHook(),
    difference_air=False,
    build_image_encoder=IdentityFunction,
    build_cell=lambda scope: tf.contrib.rnn.LSTMBlockCell(cfg.rnn_units),
    rnn_units=256,
    build_output_network=lambda scope: MLP([128], scope=scope),
    build_object_encoder=lambda scope: MLP([512, 256], scope=scope, activation_fn=tf.nn.softplus),
    build_object_decoder=lambda scope: MLP([256, 512], scope=scope, activation_fn=tf.nn.softplus),
    fixed_values=dict(),
    fixed_weights="",

    max_time_steps=3,
    object_shape=(14, 14),
    A=50,

    z_pres_prior_log_odds="Exponential(start=10000.0, end=0.000000001, decay_rate=0.1, decay_steps=3000, log=True)",
    z_pres_temperature=1.0,
    run_all_time_steps=False,
    stopping_threshold=0.99,
    per_process_gpu_memory_fraction=0.3,
    training_wheels=0.0,
    scale_prior_mean=-1.0,
    scale_prior_std=np.sqrt(0.05),
    shift_prior_mean=0.0,
    shift_prior_std=3.0,
    attr_prior_mean=0.0,
    attr_prior_std=1.0,
    kl_weight=1.0,
    pixels_per_cell=(12, 12),
    n_channels=128,
    kernel_size=1,
)

dair_config = air_config.copy(
    difference_air=True,
    build_cell=lambda scope: FeedforwardCell(MLP([512, 256]), cfg.rnn_units),
    rnn_units=256,
)

yolo_baseline_transfer_config = alg_config.copy(
    alg_name="yolo_transfer_baseline",
    build_network=baseline.YoloBaseline_Network,
    render_hook=baseline.YoloBaseline_RenderHook(),
    A=50,
    do_train=False,
    render_step=1,
    stopping_criteria="AP_at_point_25,max",
    threshold=1.0,
    build_object_encoder=lambda scope: MLP([512, 256], scope=scope),
    build_object_decoder=lambda scope: MLP([256, 512], scope=scope),
    preserve_env=False,
    curriculum=[
        dict(min_chars=n, max_chars=n, n_train=32, n_val=200, do_train=False)
        for n in range(1, 21)],
)


def yolo_air_prepare_func():
    from dps import cfg
    cfg.anchor_boxes = [cfg.tile_shape]
    cfg.count_prior_log_odds = (
        "Exp(start=1000000.0, end={}, decay_rate=0.1, "
        "decay_steps={}, log=True)".format(
            cfg.final_count_prior_log_odds, cfg.count_prior_decay_steps)
    )
    cfg.kernel_size = (cfg.kernel_size, cfg.kernel_size)


yolo_air_config = alg_config.copy(
    alg_name="yolo_air",
    build_network=yolo_air.YoloAir_Network,
    prepare_func=yolo_air_prepare_func,

    stopping_criteria="AP_at_point_5,max",
    threshold=1.0,

    render_hook=yolo_air.YoloAir_RenderHook(),

    build_backbone=core.Backbone,
    build_next_step=core.NextStep,
    build_object_encoder=lambda scope: MLP([512, 256], scope=scope),
    build_object_decoder=lambda scope: MLP([256, 512], scope=scope),

    pixels_per_cell=(12, 12),

    n_channels=128,
    n_decoder_channels=128,
    A=50,

    sequential_cfg=dict(
        on=True,
        n_lookback=1,
        build_next_step=lambda scope: MLP([100, 100], scope=scope),
    ),

    use_concrete_kl=False,
    n_final_layers=3,
    object_shape=(14, 14),

    min_yx=-0.5,
    max_yx=1.5,

    # Found through hyper parameter search
    hw_prior_mean=np.log(0.1/0.9),
    hw_prior_std=0.5,
    count_prior_decay_steps=1000,
    final_count_prior_log_odds=0.0125,
    kernel_size=1,

    training_wheels="Exp(1.0, 0.0, decay_rate=0.0, decay_steps=1000, staircase=True)",

    train_kl=True,
    train_reconstruction=True,
    noise_schedule="Exp(0.001, 0.0, 1000, 0.1)",

    curriculum=[
        dict(postprocessing="random"),
        dict(n_train=32, n_val=200, do_train=False),
    ]
)

yolo_air_transfer_config = yolo_air_config.copy(
    alg_name="yolo_air_transfer",
    min_chars=6, max_chars=10,
    load_path=0,
    curriculum=(
        [dict(postprocessing="random")] +
        [dict(min_chars=n, max_chars=n, n_train=32, n_val=200, do_train=False) for n in range(1, 21)]),
)

progression_curriculum = [
    dict(train_example_range=(0.0, 0.0001),
         val_example_range=(0.0, 0.0001),
         training_wheels="Exp(1.0, 0.0, decay_rate=0.0, decay_steps=200, staircase=True)"),
    dict(train_example_range=(0.0, 0.1),
         val_example_range=(0.0, 0.1),
         count_prior_log_odds=0.0125,),
    dict(train_example_range=(0.0, 0.9),
         val_example_range=(0.9, 1.0),
         count_prior_log_odds=0.0125,),
]


yolo_air_progression_config = yolo_air_config.copy(
    alg_name="yolo_air_progression",
    training_wheels=0,
    curriculum=progression_curriculum,
)


# --- MATH ---


def math_prepare_func():
    from dps import cfg

    yolo_air_prepare_func()

    decoder_kind = cfg.decoder_kind
    if decoder_kind == "mlp":
        cfg.build_math_network = lambda scope: MLP([256, 256, 256, 128], scope=scope)
    elif decoder_kind == "seq":
        cfg.build_math_network = yolo_math.SequentialRegressionNetwork
        cfg.build_math_cell = lambda scope: tf.contrib.rnn.LSTMBlockCell(128)
    elif decoder_kind == "attn":
        cfg.build_math_network = yolo_math.AttentionRegressionNetwork
        cfg.ar_n_filters = 256
    elif decoder_kind == "add":
        cfg.build_math_network = yolo_math.AdditionNetwork
        cfg.math_A = 10
        n_cells = (
            int(np.ceil(cfg.image_shape[0] / cfg.pixels_per_cell[0])) *
            int(np.ceil(cfg.image_shape[1] / cfg.pixels_per_cell[1])))
        cfg.largest_digit = n_cells * 9
    else:
        raise Exception("Unknown value for decoder_kind: '{}'".format(decoder_kind))


def continue_prepare_func():
    from dps import cfg
    import os

    math_prepare_func()

    repeat = int(cfg.get('repeat', 0))

    candidates = sorted(os.listdir(cfg.init_path))
    load_path = os.path.join(cfg.init_path, candidates[repeat], "weights/best_of_stage_0")

    cfg.load_path = {
        "network/reconstruction": load_path
    }


yolo_math_config = yolo_air_config.copy(
    alg_name="yolo_math",
    build_network=yolo_math.YoloAir_MathNetwork,
    stopping_criteria="math_accuracy,max",
    threshold=1.0,

    prepare_func=math_prepare_func,
    decoder_kind="seq",

    build_math_network=yolo_math.SequentialRegressionNetwork,
    build_math_cell=lambda scope: tf.contrib.rnn.LSTMBlockCell(128),
    build_math_output=lambda scope: MLP([100, 100], scope=scope),
    build_math_input=lambda scope: MLP([100, 100], scope=scope),

    math_weight=1.0,
    curriculum=[dict()],
    math_A=None,
)


def yolo_math_1stage_prepare_func():
    from dps import cfg
    import numpy as np
    math_prepare_func()
    cfg.math_weight = "Exp(0.0, 50., 10000, 0.9)"
    # cfg.math_weight = "Exp(0.0, {}, 10000, 0.9)".format(np.product(cfg.image_shape))


yolo_math_1stage_config = yolo_math_config.copy(
    prepare_func=yolo_math_1stage_prepare_func
)

curriculum_2stage = [
    dict(
        stopping_criteria="loss_reconstruction,min",
        threshold=0.0,
        math_weight=0.0,
        fixed_weights="math",
    ),
    dict(
        math_weight=1.0,
        train_reconstruction=False,
        train_kl=False,
        noisy=False,
        fixed_weights="encoder decoder object_encoder object_decoder "
                      "box attr obj backbone edge",
        load_path={"network/reconstruction": -1},
    )
]

yolo_math_2stage_config = yolo_math_config.copy(
    alg_name="yolo_math_2stage",
    curriculum=copy.deepcopy(curriculum_2stage),
)
yolo_math_2stage_config['curriculum'][0].update(
    stopping_criteria="count_error,min",
    threshold=0.0,
)

yolo_math_4stage_config = yolo_math_config.copy(
    alg_name="yolo_math_4stage",

    stopping_criteria="count_error,min",
    threshold=0.0,
    math_weight=0.0,
    fixed_weights="math",

    training_wheels=0,
    load_path={"network/reconstruction": -1},
    curriculum=progression_curriculum + [copy.deepcopy(curriculum_2stage[1])],
)
yolo_math_4stage_config['curriculum'][-1].update(
    stopping_criteria="math_accuracy,max",
)

# --- SIMPLE_MATH ---

simple_math_config = yolo_math_config.copy(
    alg_name="simple_math",
    build_network=yolo_math.SimpleMathNetwork,
    render_hook=yolo_math.SimpleMath_RenderHook(),
    build_math_encoder=core.Backbone,
    build_math_decoder=core.InverseBackbone,
    train_reconstruction=False,
    train_kl=False,
    noisy=False,
)

simple_math_2stage_config = simple_math_config.copy(
    alg_name="simple_math_2stage",
    curriculum=curriculum_2stage,
)

# --- XO ---

yolo_xo_config = yolo_math_config.copy(
    alg_name="yolo_xo",
    build_network=yolo_xo.YoloAIR_XONetwork,
    build_math_network=yolo_math.AttentionRegressionNetwork,
    balanced=True,
)

curriculum_xo_2stage = copy.deepcopy(curriculum_2stage)
curriculum_xo_2stage[0]['postprocessing'] = "random"

yolo_xo_2stage_config = yolo_xo_config.copy(
    alg_name="yolo_xo_2stage",
    curriculum=curriculum_xo_2stage,
)

yolo_xo_init_config = yolo_xo_config.copy()
yolo_xo_init_config.update(curriculum_xo_2stage[0])
yolo_xo_init_config.update(
    alg_name="yolo_xo_init",
    keep_prob=0.25,
    balanced=False,
    n_train=60000,
    curriculum=[dict()],
)

yolo_xo_continue_config = yolo_xo_config.copy()
yolo_xo_continue_config.update(curriculum_xo_2stage[1])
yolo_xo_continue_config.update(
    alg_name="yolo_xo_continue",
    prepare_func=continue_prepare_func,
    curriculum=[dict()],
    init_path="/scratch/e2crawfo/dps_data/run_experiments/GOOD_NIPS_2018/"
              "run_search_yolo-xo-init_env=xo_alg=yolo-xo-init_duration=long_seed=0_2018_06_05_09_23_55/experiments"
)

# --- SIMPLE_XO ---

simple_xo_config = simple_math_config.copy(
    alg_name="simple_xo",
    build_network=yolo_xo.SimpleXONetwork,
    build_math_network=yolo_math.AttentionRegressionNetwork,
)

simple_xo_2stage_config = simple_xo_config.copy(
    alg_name="simple_xo_2stage",
    curriculum=curriculum_2stage
)

simple_xo_init_config = simple_xo_config.copy()
simple_xo_init_config.update(curriculum_2stage[0])
simple_xo_init_config.update(
    alg_name="simple_xo_init",
    keep_prob=0.25,
    balanced=False,
    n_train=60000,
    curriculum=[dict()],
)

simple_xo_continue_config = simple_xo_config.copy()
simple_xo_continue_config.update(curriculum_2stage[1])
simple_xo_continue_config.update(
    alg_name="simple_xo_continue",
    prepare_func=continue_prepare_func,
    curriculum=[dict()],
    init_path="/scratch/e2crawfo/dps_data/run_experiments/GOOD_NIPS_2018/"
              "run_search_yolo-xo-simple-init_env=xo_alg=yolo-xo-simple-init_duration=long_seed=0_2018_06_05_09_25_02/experiments"
)
