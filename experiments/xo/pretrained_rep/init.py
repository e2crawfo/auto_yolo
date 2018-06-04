from auto_yolo import envs

readme = "xo pretrained representation experiment: pretraining the representation"

config = dict()

envs.run_experiment(
    "", config, readme, alg="yolo_xo_init", task="xo",
    durations=dict(
        long=dict(
            max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", wall_time="3hours",
            project="rpp-bengioy", cleanup_time="10mins",
            slack_time="10mins", n_repeats=6),
        med=dict(
            max_hosts=1, ppn=3, cpp=2, gpu_set="0", wall_time="30mins",
            project="rpp-bengioy", cleanup_time="10mins",
            slack_time="10mins", n_repeats=3)
    )
)