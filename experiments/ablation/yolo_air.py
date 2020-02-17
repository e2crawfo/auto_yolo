import argparse

from auto_yolo import envs
from dps.tf.updater import DummyUpdater

readme = "yolo_air ablation experiment on transfer setting"

distributions = [
    dict(min_chars=1, max_chars=5),
    dict(min_chars=6, max_chars=10),
    dict(min_chars=11, max_chars=15),
]

durations = dict(
    long=dict(
        max_hosts=1, ppn=12, cpp=2, gpu_set="0,1,2,3", wall_time="24hours",
        project="rpp-bengioy", cleanup_time="20mins",
        slack_time="5mins", n_repeats=4),

    build=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="1hour",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, step_time_limit="1hour", n_param_settings=1,
        config=dict(
            get_updater=DummyUpdater,
            render_hook=None,
            load_path=None,
            do_train=False,
            curriculum=[
                dict(min_chars=1, max_chars=5, postprocessing="random"),
                dict(min_chars=6, max_chars=10, postprocessing="random"),
                dict(min_chars=11, max_chars=15, postprocessing="random")] + [
                dict(min_chars=n, max_chars=n, n_train=32) for n in range(1, 21)])
    ),

    short=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=1, n_param_settings=4),

    small_oak=dict(
        max_hosts=1, ppn=4, cpp=2, gpu_set="0", wall_time="30mins",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=2, kind="parallel", host_pool=":"),

    build_oak=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="1year",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=1, kind="parallel", host_pool=":",
        config=dict(do_train=False)),

    oak=dict(
        max_hosts=1, ppn=4, cpp=2, gpu_set="0", wall_time="1year",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=6, kind="parallel", host_pool=":",
        step_time_limit="1year"),
)

config = dict(max_steps=int(2e5), min_chars=11, max_chars=15, patience=50000)
config["background_cfg:mode"] = "learn_solid"

parser = argparse.ArgumentParser()
parser.add_argument("--n-lookback", type=int)
args, _ = parser.parse_known_args()


# Result of parameter search.
extra_config = {
    0: {'count_prior_decay_steps': 4000,
        'final_count_prior_log_odds': 0.05,
        'hw_prior_std': 0.5,
        'kernel_size': 2},
    1: {'count_prior_decay_steps': 3000,
        'final_count_prior_log_odds': 0.0125,
        'hw_prior_std': 0.5,
        'kernel_size': 2},
    2: {'count_prior_decay_steps': 2000,
        'final_count_prior_log_odds': 0.1,
        'hw_prior_std': 0.5,
        'kernel_size': 2},
}
config.update(extra_config[args.n_lookback])
config["n_lookback"] = args.n_lookback

envs.run_experiment(
    "yolo_air_ablation_n_lookback={}".format(args.n_lookback), config, readme,
    alg="yolo_air_transfer", task="scatter", durations=durations,
    distributions=distributions
)
