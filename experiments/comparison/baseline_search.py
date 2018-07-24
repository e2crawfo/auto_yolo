from auto_yolo import envs
import argparse
import numpy as np

readme = "Searching for baseline threshold."


parser = argparse.ArgumentParser()
parser.add_argument("--n-digits", type=int, default=1)
parser.add_argument("--transfer", action="store_true")
parser.add_argument("--sc", choices="AP count_error count_1norm".split())
args, _ = parser.parse_known_args()

# dist_dict = {
#     3: np.linspace(0, .1, 101),
#     5: np.linspace(0, .1, 101),
#     7: np.linspace(.6599-0.05, .6599+0.05, 101),
#     9: np.linspace(.599-0.05, .599+0.05, 101),
# }

distributions = [dict(cc_threshold=t) for t in np.linspace(0.01, 3.0, 100)]

durations = dict(
    oak=dict(
        max_hosts=1, ppn=4, cpp=1, gpu_set="0", wall_time="1year",
        cleanup_time="1mins", slack_time="1mins", n_repeats=1, kind="parallel", host_pool=":"),
)


def build_net(scope):
    from dps.utils.tf import MLP
    return MLP([10, 10], scope=scope)


config = dict(
    curriculum=[dict()],
    render_hook=None,
    cc_threshold=0.000001,
    do_train=False,
    build_object_encoder=build_net,
    build_object_decoder=build_net
)

if args.sc == "AP":
    config.update(stopping_criteria="AP,max", threshold=1.0)
elif args.sc == "count_error":
    config.update(stopping_criteria="count_error,min", threshold=0.0)
elif args.sc == "count_1norm":
    config.update(stopping_criteria="count_1norm,min", threshold=0.0)
else:
    raise Exception()

if args.transfer:
    config["min_chars"] = args.n_digits
    config["max_chars"] = args.n_digits
    config["n_train"] = 25000
    task = "scatter"
else:
    config["min_digits"] = args.n_digits
    config["max_digits"] = args.n_digits
    config["n_train"] = 64000
    task = "arithmetic"


envs.run_experiment(
    "baseline_search_sc={}_n_digits=".format(args.sc, args.n_digits), config, readme,
    distributions=distributions, alg="baseline", durations=durations, task=task
)
