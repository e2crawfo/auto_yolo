from auto_yolo import envs
from auto_yolo.models import yolo_air

readme = "Running baseline for transfer experiment."

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
    min_digits=1, max_digits=1, do_train=True,
    render_hook=yolo_air.YoloAir_ComparisonRenderHook(show_zero_boxes=False),
)

envs.run_experiment(
    "transfer_baseline", config, readme, distributions=distributions,
    alg="baseline", task="scatter", durations=durations,
)
