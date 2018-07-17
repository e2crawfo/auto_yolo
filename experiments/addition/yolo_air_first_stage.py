from auto_yolo import envs
import numpy as np

readme = "redoing yolo_air addition experiment"

distributions = dict(
    n_train=list(1000*2**np.arange(8))
)

durations = dict(
    long=dict(
        max_hosts=1, ppn=12, cpp=2, gpu_set="0,1,2,3", wall_time="48hours",
        project="rpp-bengioy", cleanup_time="20mins",
        slack_time="5mins", n_repeats=6, step_time_limit="48hours"),

    build=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="4hours",
        project="rpp-bengioy", cleanup_time="5mins",
        slack_time="5mins", n_repeats=1, step_time_limit="4hours",
        config=dict(do_train=False)),

    short=dict(
        max_hosts=1, ppn=4, cpp=2, gpu_set="0", wall_time="30mins",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=1, config=dict(max_steps=100)),
)

config = dict(
    n_train=16000, min_digits=9, max_digits=9, largest_digit=81,
    decoder_kind="recurrent",
    stopping_criteria="AP,max",
    threshold=1.0,
    math_weight=0.0,
    fixed_weights="math",
)

envs.run_experiment(
    "yolo_addition_first_stage", config, readme,
    alg="yolo_math", task="arithmetic",
    durations=durations, distributions=distributions,
    env_kwargs=dict(ops="addition")
)
