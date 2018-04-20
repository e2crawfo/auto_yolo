import numpy as np

from dps.utils import Config
from dps.utils.tf import MLP
from dps.env.advanced import yolo_rl

# Core yolo_rl config, used as a base for all other yolo_rl configs.

yolo_rl_config = Config(
    log_name="yolo_rl",
    get_updater=yolo_rl.get_updater,

    lr_schedule=1e-4,
    batch_size=32,

    optimizer_spec="adam",
    use_gpu=True,
    gpu_allow_growth=True,
    preserve_env=True,
    stopping_criteria="TOTAL_COST,min",
    eval_mode="val",
    threshold=-np.inf,
    max_grad_norm=1.0,
    max_experiments=None,

    eval_step=1000,
    display_step=1000,
    max_steps=1e7,
    patience=10000,
    render_step=5000,

    render_hook=yolo_rl.YoloRL_RenderHook(),

    # model params

    build_backbone=yolo_rl.Backbone,
    build_next_step=yolo_rl.NextStep,
    build_object_decoder=yolo_rl.ObjectDecoder,

    use_input_attention=False,
    decoder_logit_scale=10.0,

    pixels_per_cell=(12, 12),

    anchor_boxes=[
        [7, 7],
        [7, 7]
    ],

    kernel_size=(1, 1),

    n_channels=128,
    n_decoder_channels=128,
    A=100,

    n_passthrough_features=100,

    max_hw=0.3,  # Maximum for the bounding box multiplier.
    min_hw=3.0,  # Minimum for the bounding box multiplier.

    box_std=0.0,
    attr_std=0.0,
    z_std=0.1,
    obj_exploration=0.05,
    obj_default=0.5,

    # Costs
    use_baseline=True,
    area_weight=2.,
    nonzero_weight=150.,

    local_reconstruction_cost=True,
    area_neighbourhood_size=1,
    nonzero_neighbourhood_size=1,

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
        dict(do_train=False, n_train=16, postprocessing="", preserve_env=False),  # Test on full size images.
    ],
)
