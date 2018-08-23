from auto_yolo import envs
from dps.rl.algorithms import a2c

readme = ""

config = a2c.config.copy(opt_steps_per_update=5, epsilon=0.2)

distributions = dict(
    controller_type="arn obj".split(),
    entropy_weight=[0.01, 0.02, 0.04, 0.08, 0.16, 0.32],
)


envs.run_experiment(
    "collect_search", config, readme, task="collect", distributions=distributions,
    durations=dict(
        long=dict(
            max_hosts=2, ppn=12, cpp=2, gpu_set="0,1,2,3", wall_time="12hours",
            project="rpp-bengioy", cleanup_time="10mins",
            slack_time="10mins", n_repeats=2),
        med=dict(
            max_hosts=1, ppn=3, cpp=2, gpu_set="0", wall_time="30mins",
            project="rpp-bengioy", cleanup_time="2mins",
            slack_time="2mins", n_repeats=3),
        short=dict(
            max_hosts=1, ppn=3, cpp=2, gpu_set="0", wall_time="10mins",
            project="rpp-bengioy", cleanup_time="2mins",
            slack_time="2mins", n_repeats=3)
    )
)
