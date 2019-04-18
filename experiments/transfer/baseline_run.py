from auto_yolo import envs
from auto_yolo.models import yolo_air
import argparse

readme = "Running baseline for transfer experiment."


def build_net(scope):
    from dps.utils.tf import MLP
    return MLP(n_units=[10, 10], scope=scope)


durations = dict(
    oak=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="1year",
        cleanup_time="1mins", slack_time="1mins", n_repeats=1, kind="parallel", host_pool=":"),
)

config = dict(
    curriculum=[dict()],
    n_train=32, n_val=1000, stopping_criteria="AP,max", threshold=0.99,
    min_digits=1, max_digits=1, do_train=False,
    render_hook=yolo_air.YoloAir_ComparisonRenderHook(show_zero_boxes=False),
    build_object_encoder=build_net, build_object_decoder=build_net
)

parser = argparse.ArgumentParser()
parser.add_argument("--sc", choices="AP count_error count_1norm".split())
args, _ = parser.parse_known_args()

if args.sc == "AP":
    raise Exception()
    config.update(stopping_criteria="AP,max", threshold=1.0)
    distributions = [
        dict(n_digits=1, cc_threshold=0.01),
        dict(n_digits=2, cc_threshold=0.01),
        dict(n_digits=3, cc_threshold=0.01),
        dict(n_digits=4, cc_threshold=0.01),
        dict(n_digits=5, cc_threshold=0.01),
        dict(n_digits=6, cc_threshold=0.01),
        dict(n_digits=7, cc_threshold=0.01),
        dict(n_digits=8, cc_threshold=0.01),
        dict(n_digits=9, cc_threshold=0.01),
        dict(n_digits=10, cc_threshold=0.01),
        dict(n_digits=11, cc_threshold=0.01),
        dict(n_digits=12, cc_threshold=0.01),
        dict(n_digits=13, cc_threshold=0.01),
        dict(n_digits=14, cc_threshold=0.01),
        dict(n_digits=15, cc_threshold=0.02),
        dict(n_digits=16, cc_threshold=0.17),
        dict(n_digits=17, cc_threshold=0.17),
        dict(n_digits=18, cc_threshold=0.18),
        dict(n_digits=19, cc_threshold=0.47),
        dict(n_digits=20, cc_threshold=0.47),
    ]

elif args.sc == "count_error":
    config.update(stopping_criteria="count_error,min", threshold=0.0)
    distributions = [
        dict(n_digits=1, cc_threshold=0.312),
        dict(n_digits=2, cc_threshold=0.372),
        dict(n_digits=3, cc_threshold=0.433),
        dict(n_digits=4, cc_threshold=0.372),
        dict(n_digits=5, cc_threshold=0.433),
        dict(n_digits=6, cc_threshold=0.372),
        dict(n_digits=7, cc_threshold=0.614),
        dict(n_digits=8, cc_threshold=0.493),
        dict(n_digits=9, cc_threshold=0.735),
        dict(n_digits=10, cc_threshold=0.795),
        dict(n_digits=11, cc_threshold=0.856),
        dict(n_digits=12, cc_threshold=0.886),
        dict(n_digits=13, cc_threshold=0.856),
        dict(n_digits=14, cc_threshold=0.946),
        dict(n_digits=15, cc_threshold=0.856),
        dict(n_digits=16, cc_threshold=0.976),
        dict(n_digits=17, cc_threshold=0.976),
        dict(n_digits=18, cc_threshold=1.037),
        dict(n_digits=19, cc_threshold=1.037),
        dict(n_digits=20, cc_threshold=1.037),
    ]

elif args.sc == "count_1norm":
    config.update(stopping_criteria="count_1norm,min", threshold=0.0)
    distributions = [
        dict(n_digits=1, cc_threshold=0.342),
        dict(n_digits=2, cc_threshold=0.252),
        dict(n_digits=3, cc_threshold=0.403),
        dict(n_digits=4, cc_threshold=0.372),
        dict(n_digits=5, cc_threshold=0.433),
        dict(n_digits=6, cc_threshold=0.372),
        dict(n_digits=7, cc_threshold=0.584),
        dict(n_digits=8, cc_threshold=0.584),
        dict(n_digits=9, cc_threshold=0.705),
        dict(n_digits=10, cc_threshold=0.735),
        dict(n_digits=11, cc_threshold=0.735),
        dict(n_digits=12, cc_threshold=0.825),
        dict(n_digits=13, cc_threshold=0.856),
        dict(n_digits=14, cc_threshold=0.886),
        dict(n_digits=15, cc_threshold=0.886),
        dict(n_digits=16, cc_threshold=0.916),
        dict(n_digits=17, cc_threshold=0.976),
        dict(n_digits=18, cc_threshold=1.006),
        dict(n_digits=19, cc_threshold=0.976),
        dict(n_digits=20, cc_threshold=1.006),
    ]
else:
    raise Exception()

for d in distributions:
    n_digits = d['n_digits']
    d.update(
        min_chars=n_digits,
        max_chars=n_digits
    )


envs.run_experiment(
    "transfer_baseline_sc={}".format(args.sc), config, readme,
    distributions=distributions, alg="baseline", task="scatter", durations=durations,
)
