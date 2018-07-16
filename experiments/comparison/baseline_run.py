from auto_yolo import envs
from auto_yolo.models import yolo_air

readme = "Running YOLO AIR experiment."

distributions = [
    dict(n_digits=1, cc_threshold=1e-3),
    dict(n_digits=3, cc_threshold=0.0599),
    dict(n_digits=5, cc_threshold=0.0599),
    dict(n_digits=7, cc_threshold=0.5699),
    dict(n_digits=9, cc_threshold=0.6599),
]

for d in distributions:
    n_digits = d['n_digits']
    d.update(
        min_digits=n_digits,
        max_digits=n_digits
    )


durations = dict(
    oak=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="30mins",
        cleanup_time="1mins", slack_time="1mins", n_repeats=1, kind="parallel", host_pool=":"),
)

config = dict(
    curriculum=[dict()],
    n_train=64000, run_all_time_steps=True,
    stopping_criteria="AP,max", threshold=0.99, patience=50000,
    n_digits=1,
    min_digits=1,
    max_digits=1,
    max_time_steps=1,
    render_hook=yolo_air.YoloAir_ComparisonRenderHook(),
)

envs.run_experiment(
    "baseline_run", config, readme, distributions=distributions,
    alg="baseline", task="arithmetic", durations=durations,
)
