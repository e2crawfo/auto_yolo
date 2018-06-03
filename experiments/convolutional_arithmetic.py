from auto_yolo import envs
from auto_yolo.models.yolo_math import convolutional_config as alg_config

distributions = dict(n_train=[1000, 2000, 4000, 8000, 16000, 32000])
readme = "Running sample complexity experiment with convolutional network."

config = dict(
    build_env=alg_config.build_env,
    render_step=5000,
    eval_step=1000,
    patience=5000,
    max_experiences=100000000,
    max_steps=110000,
)

envs.run_experiments(
    "sample_complexity_experiment", "arithmetic",
    alg_config, config, readme, distributions,
    dict(
        long=dict(
            max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", wall_time="6hours",
            project="rpp-bengioy", step_time_limit="2hours", cleanup_time="10mins",
            slack_time="5mins", n_repeats=6),
        med=dict(
            max_hosts=1, ppn=3, cpp=2, gpu_set="0", wall_time="1hour",
            project="rpp-bengioy", step_time_limit="25mins", cleanup_time="10mins",
            slack_time="3mins", n_param_settings=6, n_repeats=1)
    )
)
