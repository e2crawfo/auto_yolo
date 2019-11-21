import numpy as np
import tensorflow as tf

from dps import cfg
from dps.utils import Config
from dps.utils.tf import (
    MLP, IdentityFunction, FeedforwardCell, ScopedCellWrapper, AttentionalRelationNetwork, ObjectNetwork,
    GridConvNet, ConvNet,
)

from auto_yolo.models import core, simple, baseline, ground_truth, yolo_air, air, nem, networks
from auto_yolo.models.obj_kl import IndependentObjKL, SimpleObjKL, SequentialObjKL


alg_config = Config(
    get_updater=core.Updater,

    shuffle_buffer_size=10000,
    batch_size=32,
    lr_schedule=1e-4,
    optimizer_spec="adam",
    max_grad_norm=1.0,
    use_gpu=True,
    gpu_allow_growth=True,
    max_experiments=None,
    preserve_env=False,
    stopping_criteria="loss_reconstruction,min",
    threshold=-np.inf,
    load_path=-1,
    start_tensorboard=20,

    max_steps=int(3e5),
    patience=50000,
    render_step=10000,
    eval_step=1000,
    display_step=1000,

    curriculum=[dict()],

    tile_shape=(48, 48),
    n_samples_per_image=4,
    postprocessing="",

    fixed_weights="",
    fixed_values=dict(),
    no_gradient="",
    attr_prior_mean=0.0,
    attr_prior_std=1.0,
    z_prior_mean=0.0,
    z_prior_std=1.0,
    A=50,
    n_objects_per_cell=1,
    importance_temp=0.25,
    edge_resampler=False,
    eval_noisy=False,

    train_reconstruction=True,
    train_kl=True,

    reconstruction_weight=1.0,
    kl_weight=1.0,

    math_weight=0.0,
    train_math=False,
    math_A=None,

    noisy=True,

    build_background_encoder=lambda scope: MLP(n_units=[10, 10], scope=scope),
    build_background_decoder=IdentityFunction,
    max_possible_objects=None
)

simple_config = alg_config.copy(
    alg_name="simple",
    build_network=simple.SimpleVAE,
    render_hook=simple.SimpleVAE_RenderHook(),

    build_encoder=networks.Backbone,
    build_decoder=networks.InverseBackbone,
    n_channels=128,
    n_final_layers=3,
    kernel_size=1,
    pixels_per_cell=(12, 12),
)


baseline_config = alg_config.copy(
    alg_name="baseline",
    build_network=baseline.Baseline_Network,
    render_hook=baseline.Baseline_RenderHook(),
    build_object_encoder=lambda scope: MLP(n_units=[512, 256], scope=scope),
    build_object_decoder=lambda scope: MLP(n_units=[256, 512], scope=scope),
    cc_threshold=1e-3,
    object_shape=(21, 21),
)

baseline_transfer_config = baseline_config.copy(
    curriculum=[
        dict(min_chars=n, max_chars=n, n_train=32, do_train=False)
        for n in range(1, 21)],
)


ground_truth_config = baseline_config.copy(
    alg_name="ground_truth",
    object_shape=(14, 14),
    build_network=ground_truth.GroundTruth_Network,
)


def yolo_air_prepare_func():
    from dps import cfg
    cfg.anchor_box = cfg.tile_shape
    cfg.count_prior_log_odds = (
        "Exp(start=1000000.0, end={}, decay_rate=0.1, "
        "decay_steps={}, log=True)".format(
            cfg.final_count_prior_log_odds, cfg.count_prior_decay_steps)
    )


yolo_air_config = alg_config.copy(
    alg_name="yolo_air",
    build_network=yolo_air.YoloAir_Network,
    prepare_func=yolo_air_prepare_func,

    stopping_criteria="AP,max",
    threshold=1.0,

    render_hook=yolo_air.YoloAir_RenderHook(),

    build_backbone=lambda scope: GridConvNet(
        layers=[
            dict(filters=128, kernel_size=4, strides=3),
            dict(filters=128, kernel_size=4, strides=2),
            dict(filters=128, kernel_size=4, strides=2),
            dict(filters=128, kernel_size=1, strides=1),
            dict(filters=128, kernel_size=1, strides=1),
            dict(filters=128, kernel_size=1, strides=1),
        ],
        scope=scope,
    ),
    build_lateral=lambda scope: MLP(n_units=[100, 100], scope=scope),
    build_object_encoder=lambda scope: MLP(n_units=[256, 128], scope=scope),
    build_object_decoder=lambda scope: MLP(n_units=[128, 256], scope=scope),

    n_backbone_features=128,
    n_passthrough_features=0,

    n_lookback=1,
    conv_object_layer=False,

    build_obj_kl=SimpleObjKL,
    obj_concrete_temp=1.0,
    obj_temp=1.0,

    IndependentObjKL=dict(
        prior_log_odds=1.0,
    ),
    SimpleObjKL=dict(
        exp_rate=0.0125,
        # exp_rate="Exp(start=0.0125, end=.125, decay_rate=0.1, decay_steps=1000)",
    ),
    SequentialObjKL=dict(
        use_concrete_kl=False,
        count_prior_dist=None,
    ),

    object_shape=(14, 14),

    min_hw=0.0,
    max_hw=1.0,

    min_yx=-0.5,
    max_yx=1.5,

    yx_prior_mean=0.0,
    yx_prior_std=1.0,

    color_logit_scale=2.0,
    alpha_logit_scale=0.1,
    alpha_logit_bias=0.0,
    # alpha_logit_bias=5.0,

    training_wheels="Exp(1.0, 0.0, decay_rate=0.0, decay_steps=1000, staircase=True)",
    noise_schedule="Exp(0.001, 0.0, 1000, 0.1)",

    # Found through hyper parameter search
    hw_prior_mean=np.log(0.1/0.9),
    hw_prior_std=0.5,
    count_prior_decay_steps=1000,
    final_count_prior_log_odds=0.0125,
)

yolo_air_conv_config = yolo_air_config.copy(
    conv_object_layer=True,
    build_conv_lateral=lambda scope: ConvNet(
        scope=scope, layers=[
            dict(filters=None, kernel_size=1, strides=1, padding="SAME"),
            dict(filters=None, kernel_size=1, strides=1, padding="SAME"),
        ],
    ),
)

yolo_air_transfer_config = yolo_air_config.copy(
    alg_name="yolo_air_transfer",
    min_chars=6, max_chars=10,
    load_path=0,
    curriculum=(
        [dict(postprocessing="random")]
        + [dict(min_chars=n, max_chars=n, n_train=32, do_train=False) for n in range(1, 21)]),
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

# Original config from tf-attend_infer_repeat
air_config = alg_config.copy(
    alg_name="attend_infer_repeat",
    build_network=air.AIR_Network,
    batch_size=64,
    render_hook=air.AIR_RenderHook(),
    difference_air=False,
    build_image_encoder=IdentityFunction,
    build_cell=lambda scope: ScopedCellWrapper(
        tf.contrib.rnn.LSTMBlockCell(cfg.rnn_n_units)),
    rnn_n_units=256,
    build_output_network=lambda scope: MLP(n_units=[128], scope=scope),
    build_object_encoder=lambda scope: MLP(n_units=[512, 256], scope=scope, fc_kwargs=dict(activation_fn=tf.nn.softplus)),
    build_object_decoder=lambda scope: MLP(n_units=[256, 512], scope=scope, fc_kwargs=dict(activation_fn=tf.nn.softplus)),

    stopping_criteria="AP,max",
    threshold=1.0,

    max_time_steps=3,
    object_shape=(14, 14),

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

    complete_rnn_input=False,
)

dair_config = air_config.copy(
    difference_air=True,
    build_cell=lambda scope: FeedforwardCell(MLP(n_units=[256, 256, 256, 256]), cfg.rnn_n_units),
)

nem_config = alg_config.copy(
    alg_name="nem",
    build_network=nem.NEM_Network,
    batch_size=16,
    lr_schedule=0.001,
    max_grad_norm=None,

    threshold=-np.inf,
    max_experiments=None,
    render_hook=nem.NeuralEM_RenderHook(4),
    render_step=5000,

    # ------- from nem.py --------

    noise_prob=0.2,                              # probability of annihilating the pixel

    # ------- from nem_model.py ------

    # general
    binary=False,
    gradient_gamma=True,       # whether to back-propagate a gradient through gamma

    # loss
    inter_weight=1.0,          # weight for the inter-cluster loss
    loss_step_weights='last',  # all, last, or list of weights
    pixel_prior=dict(
        p=0.0,                     # probability of success for pixel prior Bernoulli
        mu=0.0,                    # mean of pixel prior Gaussian
        sigma=0.25,                 # std of pixel prior Gaussian
    ),

    # em
    k=4,                       # number of components
    n_steps=10,                # number of (RN)N-EM steps
    e_sigma=0.25,              # sigma used in the e-step when pixel distributions are Gaussian (acts as a temperature)
    pred_init=0.0,             # initial prediction used to compute the input

    # ------- from network.py ------

    use_NEM_formulation=False,

    # input_network=[],
    # recurrent_network=[
    #     {'name': 'rnn', 'size': 250, 'act': 'sigmoid', 'ln': False}
    # ],
    # output_network=[
    #     {'name': 'fc', 'size': 3 * 784, 'act': 'sigmoid', 'ln': False},
    #     # {'name': 'fc', 'size': 784, 'act': '*', 'ln': False},
    # ],

    build_input_network=lambda scope: nem.FeedforwardNetwork(
        [{'name': 'input_norm'},
         {'name': 'reshape', 'shape': (48, 48, 3)},
         {'name': 'conv', 'size': 32, 'act': 'elu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True},
         {'name': 'conv', 'size': 64, 'act': 'elu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True},
         {'name': 'conv', 'size': 128, 'act': 'elu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True},
         {'name': 'reshape', 'shape': -1},
         {'name': 'fc', 'size': 512, 'act': 'elu', 'ln': True}], scope=scope),

    build_cell=lambda scope: tf.contrib.rnn.LSTMBlockCell(128),
    # {'name': 'rnn', 'size': 250, 'act': 'sigmoid', 'ln': True}

    build_output_network=lambda scope: nem.FeedforwardNetwork(
        [{'name': 'fc', 'size': 512, 'act': 'relu', 'ln': True},
         {'name': 'fc', 'size': 6 * 6 * 128, 'act': 'relu', 'ln': True},
         {'name': 'reshape', 'shape': (6, 6, 128)},
         {'name': 't_conv', 'size': 64, 'act': 'relu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True},
         {'name': 't_conv', 'size': 32, 'act': 'relu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True},
         {'name': 't_conv', 'size': 3, 'act': 'sigmoid', 'stride': [2, 2], 'kernel': (4, 4), 'ln': False},
         {'name': 'reshape', 'shape': -1}], scope=scope)
)


# --- MATH ---


def math_prepare_func():
    from dps import cfg

    decoder_kind = cfg.decoder_kind
    if decoder_kind == "mlp":
        cfg.build_math_network = lambda scope: MLP(n_units=[256, 256, 256, 128], scope=scope)
    elif decoder_kind == "recurrent":
        cfg.build_math_network = networks.SimpleRecurrentRegressionNetwork
        cfg.build_math_cell = lambda scope: tf.contrib.rnn.LSTMBlockCell(cfg.n_recurrent_units)
    elif decoder_kind == "seq":
        cfg.build_math_network = networks.SequentialRegressionNetwork
        cfg.build_math_cell = lambda scope: tf.contrib.rnn.LSTMBlockCell(cfg.n_recurrent_units)
    elif decoder_kind == "attn":
        cfg.build_math_network = networks.AttentionRegressionNetwork
        cfg.ar_n_filters = 256
    elif decoder_kind == "obj":
        cfg.build_math_network = ObjectNetwork
        cfg.n_repeats = 1
    elif decoder_kind == "arn":
        cfg.build_math_network = AttentionalRelationNetwork
        cfg.n_repeats = 2
    elif decoder_kind == "add":
        cfg.build_math_network = networks.AdditionNetwork
        cfg.math_A = 10
        n_cells = (
            int(np.ceil(cfg.image_shape[0] / cfg.pixels_per_cell[0]))
            * int(np.ceil(cfg.image_shape[1] / cfg.pixels_per_cell[1])))
        cfg.largest_digit = n_cells * 9
    else:
        raise Exception("Unknown value for decoder_kind: '{}'".format(decoder_kind))

    if "stage1_path" in cfg:
        import os, glob  # noqa
        pattern = os.path.join(
            cfg.stage1_path, "experiments",
            "*idx={}*repeat={}*".format(cfg.idx, cfg.repeat))
        paths = glob.glob(pattern)
        assert len(paths) == 1
        cfg.load_path = {"network/representation": os.path.join(paths[0], "weights", "best_of_stage_0")}


math_config = Config(
    prepare_func=math_prepare_func,
    build_math_input=lambda scope: MLP(n_units=[100, 100], scope=scope),
    build_math_output=lambda scope: MLP(n_units=[100, 100], scope=scope),
    train_math=True,
    math_weight=1.0,
    stopping_criteria="math_accuracy,max",
    threshold=1.0,
    decoder_kind="recurrent",
    patience=10000,

    train_kl=False,
    train_reconstruction=False,
    noisy=False,

    n_recurrent_units=128,

    # for obj and arn
    d=128,
    layer_norm=True,
    symmetric_op="max",
    use_mask=True,

    # For obj
    # build_on_input_network=lambda scope: MLP(n_units=[128, 128], scope=scope),
    # build_on_object_network=lambda scope: MLP(n_units=[128, 128], scope=scope),
    # build_on_output_network=lambda scope: MLP(n_units=[128, 128, 128], scope=scope),
    build_on_input_network=lambda scope: MLP(n_units=[128], scope=scope),
    build_on_object_network=lambda scope: MLP(n_units=[128], scope=scope),
    build_on_output_network=lambda scope: MLP(n_units=[128, 128], scope=scope),

    # For arn
    build_arn_network=lambda scope: MLP(n_units=[128, 128], scope=scope),
    build_arn_object_network=lambda scope: MLP(n_units=[128, 128], scope=scope),
    n_heads=1,
)

simple_math_config = simple_config.copy(
    math_config,
    alg_name="simple_math",
)

baseline_math_config = baseline_config.copy(
    math_config,
    alg_name="baseline_math",
)

ground_truth_math_config = ground_truth_config.copy(
    math_config,
    alg_name="ground_truth_math",
)

yolo_air_math_config = yolo_air_config.copy(
    math_config,
    alg_name="yolo_air_math",
)
yolo_air_math_config.prepare_func = [math_config.prepare_func, yolo_air_config.prepare_func]

air_math_config = air_config.copy(
    math_config,
    alg_name="air_math",
)

# def continue_prepare_func():
#     from dps import cfg
#     import os
#
#     math_prepare_func()
#
#     repeat = int(cfg.get('repeat', 0))
#
#     candidates = sorted(os.listdir(cfg.init_path))
#     load_path = os.path.join(cfg.init_path, candidates[repeat], "weights/best_of_stage_0")
#
#     cfg.load_path = {
#         "network/reconstruction": load_path
#     }
#
#
# def math_1stage_prepare_func():
#     from dps import cfg
#     import numpy as np
#     math_prepare_func()
#     cfg.math_weight = "Exp(0.0, 50., 10000, 0.9)"
#     # cfg.math_weight = "Exp(0.0, {}, 10000, 0.9)".format(np.product(cfg.image_shape))
#
#
# math_1stage_config = math_config.copy(
#     prepare_func=math_1stage_prepare_func
# )
#
# math_4stage_config = math_config.copy(
#     alg_name="math_4stage",
#
#     stopping_criteria="count_error,min",
#     threshold=0.0,
#     math_weight=0.0,
#     fixed_weights="math",
#
#     training_wheels=0,
#     load_path={"network/reconstruction": -1},
#     curriculum=progression_curriculum + [copy.deepcopy(curriculum_2stage[1])],
# )
# math_4stage_config['curriculum'][-1].update(
#     stopping_criteria="math_accuracy,max",
# )
#
# # --- SIMPLE_MATH ---
#
# simple_math_config = math_config.copy(
#     alg_name="simple_math",
#     build_network=networks.SimpleMathNetwork,
#     render_hook=networks.SimpleMath_RenderHook(),
#     build_math_encoder=networks.Backbone,
#     build_math_decoder=networks.InverseBackbone,
#     train_reconstruction=False,
#     train_kl=False,
#     noisy=False,
# )
#
# simple_math_2stage_config = simple_math_config.copy(
#     alg_name="simple_math_2stage",
#     curriculum=curriculum_2stage,
# )
#
# # --- XO ---
#
# yolo_xo_config = math_config.copy(
#     alg_name="yolo_xo",
#     build_network=yolo_xo.YoloAIR_XONetwork,
#     build_math_network=networks.AttentionRegressionNetwork,
#     balanced=True,
# )
#
# curriculum_xo_2stage = copy.deepcopy(curriculum_2stage)
# curriculum_xo_2stage[0]['postprocessing'] = "random"
#
# yolo_xo_2stage_config = yolo_xo_config.copy(
#     alg_name="yolo_xo_2stage",
#     curriculum=curriculum_xo_2stage,
# )
#
# yolo_xo_init_config = yolo_xo_config.copy()
# yolo_xo_init_config.update(curriculum_xo_2stage[0])
# yolo_xo_init_config.update(
#     alg_name="yolo_xo_init",
#     keep_prob=0.25,
#     balanced=False,
#     n_train=60000,
# )
#
# yolo_xo_continue_config = yolo_xo_config.copy()
# yolo_xo_continue_config.update(curriculum_xo_2stage[1])
# yolo_xo_continue_config.update(
#     alg_name="yolo_xo_continue",
#     prepare_func=continue_prepare_func,
#     init_path="/scratch/e2crawfo/dps_data/run_experiments/GOOD_NIPS_2018/"
#               "run_search_yolo-xo-init_env=xo_alg=yolo-xo-init_duration=long_seed=0_2018_06_05_09_23_55/experiments"
# )
#
# # --- SIMPLE_XO ---
#
# simple_xo_config = simple_math_config.copy(
#     alg_name="simple_xo",
#     build_network=yolo_xo.SimpleXONetwork,
#     build_math_network=networks.AttentionRegressionNetwork,
# )
#
# simple_xo_2stage_config = simple_xo_config.copy(
#     alg_name="simple_xo_2stage",
#     curriculum=curriculum_2stage
# )
#
# simple_xo_init_config = simple_xo_config.copy()
# simple_xo_init_config.update(curriculum_2stage[0])
# simple_xo_init_config.update(
#     alg_name="simple_xo_init",
#     keep_prob=0.25,
#     balanced=False,
#     n_train=60000,
# )
#
# simple_xo_continue_config = simple_xo_config.copy()
# simple_xo_continue_config.update(curriculum_2stage[1])
# simple_xo_continue_config.update(
#     alg_name="simple_xo_continue",
#     prepare_func=continue_prepare_func,
#     init_path="/scratch/e2crawfo/dps_data/run_experiments/GOOD_NIPS_2018/"
#               "run_search_yolo-xo-simple-init_env=xo_alg=yolo-xo-simple-init_duration=long_seed=0_2018_06_05_09_25_02/experiments"
# )
