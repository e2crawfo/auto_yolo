import numpy as np

from dps.utils import Config
from dps.utils.tf import MLP
from dps.env.advanced import yolo_rl, air

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
    eval_mode="val",
    threshold=-np.inf,
    max_experiments=None,

    render_hook=yolo_rl.YoloRL_RenderHook(),

    # model params

    build_backbone=yolo_rl.NewBackbone,
    build_next_step=yolo_rl.NextStep,
    build_object_decoder=yolo_rl.ObjectDecoder,

    use_input_attention=False,
    decoder_logit_scale=10.0,

    max_object_shape=(28, 28),
    pixels_per_cell=(8, 8),

    anchor_boxes=[[14, 14]],

    kernel_size=(1, 1),

    n_channels=128,
    n_decoder_channels=128,
    A=100,
    n_passthrough_features=0,
    n_backbone_features=100,

    min_hw=0.3,
    max_hw=3.0,

    box_std=0.0,
    attr_std=0.0,
    z_std=0.1,
    obj_exploration=0.05,
    obj_default=0.5,
    explore_during_val=False,

    # Costs
    use_baseline=True,
    area_weight=0.01,
    nonzero_weight=1.0,
    rl_weight=1.0,

    local_reconstruction_cost=True,
    area_neighbourhood_size=1,
    nonzero_neighbourhood_size=1,
    target_area=0.,

    fixed_values=dict(),
    fixed_weights="",
    order="box obj z attr",

    sequential_cfg=dict(
        on=True,
        lookback_shape=(2, 2, 2),
        build_next_step=lambda scope: MLP([100, 100], scope=scope),
    ),

    curriculum=[
        dict(obj_exploration=0.2),
        dict(obj_exploration=0.1),
        dict(obj_exploration=0.05),
        dict(obj_exploration=0.03),
        dict(obj_exploration=0.02),
        dict(obj_exploration=0.01),
        dict(do_train=False, n_train=16, min_chars=1, postprocessing="", preserve_env=False),
        dict(postprocessing="", preserve_env=False),
    ],
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
