from auto_yolo import envs
import numpy as np
import argparse

readme = "Testing yolo_air."

distributions = dict(
    hw_prior_mean=[np.log(0.1/0.9), np.log(0.15/0.85), np.log(0.2/0.8), np.log(0.25/0.75)],
    count_prior_decay_steps=[3000, 6000, 9000, 12000],
    final_count_prior_log_odds=[0.0125, 0.025, 0.05, 0.1],
)

durations = dict(
    long=dict(
        max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", wall_time="6hours",
        project="rpp-bengioy", cleanup_time="20mins",
        slack_time="5mins", n_repeats=6),

    build=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="2hours",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, step_time_limit="2hours",
        config=dict(do_train=False)),

    short=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=1, n_param_settings=4),

    oak=dict(
        host_pool=[":"], kind="parallel",
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="1hour",
        project="rpp-bengioy", cleanup_time="1mins", slack_time="1mins",
        step_time_limit="1hour", n_repeats=10, n_param_settings=1,
        config=dict(max_steps=4000)),
)

parser = argparse.ArgumentParser()
parser.add_argument("--game")
args, _ = parser.parse_known_args()

config = dict(
    game=args.game,
    postprocessing="",
    curriculum=[
        dict(fixed_weights="backbone box obj z object_encoder object_decoder", postprocessing="random", max_steps=10000),
        dict(postprocessing="random"),
        dict(do_train=False),
        # dict(fixed_weights="backbone box obj z object_encoder object_decoder", postprocessing="random", max_steps=2000),
        # dict(postprocessing="random", load_path="/media/data/dps_data/logs/atari_env=task=atari/exp_alg=yolo-air_seed=723605649_2018_09_04_15_38_54/weights/best_of_stage_0")
        #dict(do_train=False, load_path="/media/data/dps_data/logs/atari_env=task=atari/exp_alg=yolo-air_seed=1742161367_2018_09_04_16_03_06/weights/best_of_stage_0"),
    ],

    # Experimental
    hw_prior_mean=np.log(0.1/0.9),
    hw_prior_std=0.5,
    count_prior_decay_steps=3000,
    final_count_prior_log_odds=0.1,
    kernel_size=2,
    training_wheels="Exp(1.0, 0.0, decay_rate=0.0, decay_steps=4000, staircase=True)",

    render_step=1000,
    stopping_criteria="loss_reconstruction,min",
)

envs.run_experiment(
    "atari_game={}".format(args.game), config, readme, alg="yolo_air",
    task="atari", durations=durations, distributions=distributions,
)
