from dps.utils import Config

from auto_yolo import envs

readme = "redoing yolo_air transfer experiment"

config = Config(
    min_chars=6, max_chars=10,
    curriculum=[dict(postprocessing="random")] + [dict(min_chars=n, max_chars=n) for n in range(1, 21)],
    tile_shape=(48, 48),
    postprocessing="",
    n_samples_per_image=4,
    preserve_env=False,
)

durations = dict(
    long=dict(
        max_hosts=1, ppn=4, cpp=2, gpu_set="0,1", wall_time="24hours",
        project="rpp-bengioy", cleanup_time="20mins",
        slack_time="5mins", n_repeats=8, step_time_limit="24hours"),
    build=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="2hours",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, step_time_limit="2hours"),
    short=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="4hours",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, step_time_limit="4hours"),
)

envs.run_experiment(
    "yolo_air_transfer", config, readme, alg="yolo_air", task="scatter", durations=durations,
)
