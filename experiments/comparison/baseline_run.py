from auto_yolo import envs
from auto_yolo.models import yolo_air

readme = "Running baseline for comparison experiment."

distributions = [
    dict(n_digits=1, cc_threshold=0.01),
    dict(n_digits=3, cc_threshold=0.01),
    dict(n_digits=5, cc_threshold=0.02),
    dict(n_digits=7, cc_threshold=0.6),
    dict(n_digits=9, cc_threshold=0.52),
]

for d in distributions:
    n_digits = d['n_digits']
    d.update(
        min_digits=n_digits,
        max_digits=n_digits
    )


durations = dict(
    oak=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="30mins",
        cleanup_time="1mins", slack_time="1mins", n_repeats=1, kind="parallel", host_pool=":"),
)

config = dict(
    curriculum=[dict()],
    n_train=32, n_val=1000, stopping_criteria="AP,max", threshold=0.99,
    min_digits=1, max_digits=1, do_train=False,
    render_hook=yolo_air.YoloAir_ComparisonRenderHook(show_zero_boxes=False),
)

envs.run_experiment(
    "comparison_baseline", config, readme, distributions=distributions,
    alg="baseline", task="arithmetic", durations=durations,
)
