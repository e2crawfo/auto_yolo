from auto_yolo import envs
from auto_yolo.algs import yolo_air_config

distributions = dict(n_train=[1000, 2000, 4000, 8000, 16000, 32000])
readme = "Testing sample complexity of yolo_air."

envs.run_experiments(
    "sample_complexity_experiment", "arithmetic",
    yolo_air_config, config, readme, distributions,
    dict(
        long=dict(
            max_hosts=3, ppn=12, cpp=2, gpu_set="0,1,2,3", wall_time="6hours",
            project="rpp-bengioy", cleanup_time="15mins", slack_time="15mins",
            n_repeats=6),
        med=dict(
            max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", wall_time="1hour",
            project="rpp-bengioy", cleanup_time="10mins", slack_time="10mins",
            n_param_settings=3, n_repeats=2)
    )
)
