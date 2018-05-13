import numpy as np

from dps.utils import Config
from dps.utils.tf import MLP
from dps.env.advanced import yolo_rl, air, yolo_air

# Core yolo_rl config, used as a base for all other yolo_rl configs.

yolo_rl_config = Config(
    log_name="yolo_rl",
    get_updater=yolo_rl.get_updater,

    batch_size=32,
    lr_schedule=1e-4,
    optimizer_spec="adam",
    max_grad_norm=1.0,

    use_gpu=True,
    gpu_allow_growth=True,

    preserve_env=True,
    stopping_criteria="TOTAL_COST,min",
    threshold=-np.inf,
    eval_mode="val",
    max_experiments=None,

    render_hook=yolo_rl.YoloRL_RenderHook(),

    # model params

    build_backbone=yolo_rl.Backbone,
    build_next_step=yolo_rl.NextStep,
    build_object_encoder=lambda scope: MLP([100, 100], scope=scope),
    build_object_decoder=lambda scope: MLP([100, 100], scope=scope),
    # build_object_decoder=yolo_rl.ObjectDecoder,

    use_input_attention=False,
    decoder_logit_scale=10.0,

    max_object_shape=(28, 28),
    pixels_per_cell=(8, 8),
    anchor_boxes=[[14, 14]],

    kernel_size=(3, 3),

    n_channels=128,
    n_decoder_channels=128,
    A=100,
    n_passthrough_features=0,
    n_backbone_features=100,

    min_hw=0.0,
    max_hw=3.0,

    box_std=0.0,
    attr_std=0.0,
    z_std=0.1,
    obj_exploration=0.05,
    obj_default=0.5,
    explore_during_val=False,

    use_baseline=True,

    # Costs
    area_weight=1.0,
    nonzero_weight=25.0,
    rl_weight=1.0,
    hw_weight=None,
    z_weight=None,

    area_neighbourhood_size=2,
    hw_neighbourhood_size=None,
    nonzero_neighbourhood_size=2,
    local_reconstruction_cost=True,

    target_area=0.0,
    target_hw=0.0,

    fixed_values=dict(),
    fixed_weights="",
    order="box obj z attr",

    sequential_cfg=dict(
        on=True,
        lookback_shape=(2, 2, 2),
        build_next_step=lambda scope: MLP([100, 100], scope=scope),
    ),

    # curriculum=[
    #     dict(max_steps=5000, rl_weight=None, area_weight=None, fixed_values=dict(obj=1.0)),
    #     dict(max_steps=5000, rl_weight=None, fixed_values=dict(obj=1.0)),
    #     dict(obj_exploration=0.2),
    #     dict(obj_exploration=0.1),
    #     dict(obj_exploration=0.05),
    #     dict(do_train=False, n_train=16, min_chars=1, postprocessing="", preserve_env=False),
    #     dict(obj_exploration=0.05, preserve_env=False, patience=10000000),
    # ],
)

# Core air config, used as a base for all other air configs.

air_config = Config(
    log_name="attend_infer_repeat",
    get_updater=air.get_updater,

    # training loop params

    batch_size=64,
    lr_schedule=1e-4,
    optimizer_spec="adam",
    max_grad_norm=1.0,

    use_gpu=True,
    gpu_allow_growth=True,

    preserve_env=True,
    stopping_criteria="loss,min",
    eval_mode="val",
    threshold=-np.inf,
    max_experiments=None,

    verbose_summaries=False,

    render_hook=air.AIR_RenderHook(),

    # model params - based on the values in tf-attend-infer-repeat/training.py
    max_time_steps=3,
    rnn_units=256,
    object_shape=(14, 14),
    vae_latent_dimensions=50,
    vae_recognition_units=(512, 256),
    vae_generative_units=(256, 512),
    scale_prior_mean=-1.0,
    scale_prior_variance=0.05,
    shift_prior_mean=0.0,
    shift_prior_variance=1.0,
    vae_prior_mean=0.0,
    vae_prior_variance=1.0,
    # vae_likelihood_std=0.0,
    vae_likelihood_std=0.3,  # <- an odd value...maybe comes from the output gaussian with std 0.3 used in original paper?
    scale_hidden_units=64,
    shift_hidden_units=64,
    z_pres_hidden_units=64,
    z_pres_prior_log_odds="Exponential(start=10000.0, end=0.000000001, decay_rate=0.1, decay_steps=3000, log=True)",
    z_pres_temperature=1.0,
    stopping_threshold=0.99,
    cnn=False,
    cnn_filters=128,
    per_process_gpu_memory_fraction=0.3,
)


def yolo_air_prepare_func():
    from dps import cfg
    if cfg.postprocessing is not None:
        cfg.anchor_boxes = [cfg.tile_shape]
    else:
        cfg.anchor_boxes = [cfg.image_shape]

    cfg.count_prior_log_odds = (
        "Exp(start=10000.0, end={}, decay_rate=0.1, "
        "decay_steps=1000, log=True)".format(cfg.final_count_prior_log_odds)
    )
    cfg.kernel_size = (cfg.kernel_size, cfg.kernel_size)


yolo_air_config = Config(
    log_name="yolo_air",
    get_updater=yolo_air.get_updater,
    prepare_func=yolo_air_prepare_func,

    max_steps=50000,
    patience=100000,

    render_hook=yolo_air.YoloAir_RenderHook(),

    build_backbone=yolo_rl.Backbone,
    build_next_step=yolo_rl.NextStep,
    build_object_encoder=lambda scope: MLP([512, 256], scope=scope),
    build_object_decoder=lambda scope: MLP([256, 512], scope=scope),
    # build_backbone=yolo_rl.NewBackbone,
    # max_object_shape=(28, 28),
    # build_object_decoder=ObjectDecoder,

    pixels_per_cell=(12, 12),

    kernel_size=1,

    n_channels=128,
    n_decoder_channels=128,
    A=50,

    sequential_cfg=dict(
        on=True,
        lookback_shape=(2, 2, 2),
        build_next_step=lambda scope: MLP([100, 100], scope=scope),
    ),

    hw_prior_mean=np.log(0.1/0.9),
    hw_prior_std=1.0,
    final_count_prior_log_odds=0.05,

    use_concrete_kl=False,
    n_final_layers=3,

    curriculum=[
        dict(),
        dict(do_train=False, n_train=16, min_chars=1, postprocessing="", preserve_env=False),
    ],
)
