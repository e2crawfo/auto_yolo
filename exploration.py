import clify
from dps.env.advanced import yolo_rl
from dps.datasets import EMNIST_ObjectDetection


def prepare_func():
    from dps import cfg
    cfg.kernel_size = (cfg.kernel_size, cfg.kernel_size)


distributions = dict(
    area_weight=list([0.25, 0.5, 1.0, 2.0]),
    nonzero_weight=list([50, 100, 150, 200]),
    box_std=[0.0, 0.1],
    kernel_size=[1, 3],
)

config = yolo_rl.config.copy(
    prepare_func=prepare_func,
    render_step=100000,
    eval_step=1000,
    patience=10000,
    area_weight=None,
    nonzero_weight=None,

    # TODO
    max_steps=2000,
)

# Create the datasets if necessary.
print("Forcing creation of first dataset.")
with config:
    train = EMNIST_ObjectDetection(n_examples=int(config.n_train), shuffle=True, example_range=(0.0, 0.9))
    val = EMNIST_ObjectDetection(n_examples=int(config.n_val), shuffle=True, example_range=(0.9, 1.))

print("Forcing creation of second dataset.")
with config.copy(config.curriculum[-1]):
    train = EMNIST_ObjectDetection(n_examples=int(config.n_train), shuffle=True, example_range=(0.0, 0.9))
    val = EMNIST_ObjectDetection(n_examples=int(config.n_val), shuffle=True, example_range=(0.9, 1.))


from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(config=config, distributions=distributions)
