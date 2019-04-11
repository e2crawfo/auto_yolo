import argparse

from auto_yolo import envs
from auto_yolo.models import yolo_air

readme = "Searching for YOLO_AIR parameters."

durations = dict(
    long=dict(
        max_hosts=1, ppn=16, cpp=1, gpu_set="0,1,2,3", wall_time="27hours",
        project="rpp-bengioy", cleanup_time="10mins",
        slack_time="10mins", n_repeats=3, n_param_settings=3*16),

    build=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, step_time_limit="20mins", n_param_settings=1,
        config=dict(do_train=False)),

    short=dict(
        max_hosts=1, ppn=3, cpp=2, gpu_set="0", wall_time="30mins",
        project="rpp-bengioy", cleanup_time="2mins", n_param_settings=3,
        slack_time="2mins", n_repeats=1, config=dict(max_steps=100))
)

parser = argparse.ArgumentParser()
parser.add_argument("--n-lookback", type=int, default=1)
args, _ = parser.parse_known_args()

distributions = dict(
    kernel_size=[1, 2],
    hw_prior_std=[0.5, 1.0, 2.0],  # Anything outside of these bounds doesn't work very well.
    final_count_prior_log_odds=[0.1, 0.05, 0.025, 0.0125],
    count_prior_decay_steps=[1000, 2000, 3000, 4000],
)

if not args.n_lookback:
    distributions["kernel_size"] = [1, 2, 3]

config = dict(
    curriculum=[dict()],
    n_train=64000, stopping_criteria="AP,max", threshold=0.99, patience=50000,
    render_hook=yolo_air.YoloAir_RenderHook(),
    min_digits=1, max_digits=9, max_steps=2e5,
    background_cfg=dict(mode="learn_solid"),
    train_example_range=(0.0, 0.7),
    val_example_range=(0.7, 0.8),
    test_example_range=(0.8, 0.9),
)
config["n_lookback"] = args.n_lookback


envs.run_experiment(
    "yolo_air_search_n_lookback={}".format(args.n_lookback), config, readme,
    distributions=distributions, alg="yolo_air", task="arithmetic", durations=durations,
)
