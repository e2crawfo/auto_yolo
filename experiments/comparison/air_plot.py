from auto_yolo import envs
from auto_yolo.models import air

readme = "Plotting results of AIR experiment."

weight_path = (
    "/media/data/Dropbox/experiment_data/active/nips2018/CEDAR/comparison/"
    "run_search_air-run_env=size=14-in-colour=False-task=arithmetic-ops="
    "addition_alg=attend-infer-repeat_duration=long_seed=0_2018_07_10_09_08_58/"
    "experiments/{}/weights/best_of_stage_0"
)


distributions = [
    {'n_digits': 1,
     'rnn_n_units': 256,
     'scale_prior_mean': -2.9444389791664403,
     'scale_prior_std': 0.2,
     'shift_prior_std': 1.0,
     'load_path': weight_path.format("exp_idx=0_repeat=0_seed=258868792_2018_07_10_11_59_43")
    },

    {'n_digits': 3,
     'rnn_n_units': 256,
     'scale_prior_mean': -2.197224577336219,
     'scale_prior_std': 0.2,
     'shift_prior_std': 1.0,
     'load_path': weight_path.format("exp_idx=1_repeat=0_seed=1768564497_2018_07_10_11_59_43")
    },

    {'n_digits': 5,
     'rnn_n_units': 256,
     'scale_prior_mean': -2.197224577336219,
     'scale_prior_std': 0.2,
     'shift_prior_std': 2.0,
     'load_path': weight_path.format("exp_idx=2_repeat=0_seed=2063514704_2018_07_10_11_59_43")
    },

    {'n_digits': 7,
     'rnn_n_units': 256,
     'scale_prior_mean': -1.3862943611198906,
     'scale_prior_std': 0.4,
     'shift_prior_std': 4.0,
     'load_path': weight_path.format("exp_idx=3_repeat=0_seed=1230766506_2018_07_10_23_56_01")
    },

    {'n_digits': 9,
     'rnn_n_units': 256,
     'scale_prior_mean': -0.4054651081081643,
     'scale_prior_std': 0.2,
     'shift_prior_std': 1.0,
     'load_path': weight_path.format("exp_idx=4_repeat=0_seed=1636936418_2018_07_10_23_56_01")
    },
]


for dist in distributions:
    n_digits = dist['n_digits']
    dist.update(
        min_digits=n_digits,
        max_digits=n_digits,
        max_time_steps=n_digits)


durations = dict(
    oak=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="30mins",
        cleanup_time="1mins", slack_time="1mins", n_repeats=1, kind="parallel", host_pool=":"),
)

config = dict(
    curriculum=[dict()], do_train=False,
    n_train=64000, run_all_time_steps=True,
    stopping_criteria="AP,max", threshold=0.99, patience=50000,
    render_hook=air.AIR_ComparisonRenderHook(),
)


envs.run_experiment(
    "air_run", config, readme, distributions=distributions,
    alg="air", task="arithmetic", durations=durations,
)
