from auto_yolo import envs
from dps.updater import DummyUpdater

readme = "Running first stage (representation learning) for yolo_air on addition task."

distributions = [dict(n_train=1000 * 2**i) for i in range(8)]

durations = dict(
    long=dict(
        max_hosts=1, ppn=12, cpp=2, gpu_set="0,1,2,3", project="rpp-bengioy",
        wall_time="48hours", cleanup_time="5mins", slack_time="5mins",
        n_repeats=6, step_time_limit="48hours"),

    build=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="2hours",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, step_time_limit="2hours",
        host_pool=[":"], kind="parallel",
        config=dict(
            get_updater=DummyUpdater,
            render_hook=None,
            load_path=None,
            do_train=False,
            curriculum=distributions
        )
    ),

    short=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=1, n_param_settings=4),
)

config = dict(max_steps=2e5, patience=50000, n_train=16000)


if __name__ == "__main__":
    envs.run_experiment(
        "addition-stage1", config, readme, alg="yolo_air",
        task="arithmetic2", durations=durations, distributions=distributions
    )
