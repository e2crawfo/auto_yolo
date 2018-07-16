from auto_yolo import envs
from auto_yolo.models import yolo_air

readme = "Plotting results of YOLO_AIR experiment."

weight_path = (
    "/media/data/Dropbox/experiment_data/active/nips2018/CEDAR/comparison/"
    "run_search_resubmit_seed=0_2018_07_12_11_17_08/"
    "experiments/{}/weights/best_of_stage_0"
)


distributions = [
    {'n_digits': 1,
     'load_path': weight_path.format("exp_idx=0_repeat=0_seed=919310262_2018_07_10_15_54_09")
    },

    {'n_digits': 3,
     'load_path': weight_path.format("exp_idx=1_repeat=0_seed=1501865814_2018_07_10_15_54_09")
    },

    {'n_digits': 5,
     'load_path': weight_path.format("exp_idx=2_repeat=0_seed=1730811461_2018_07_10_15_54_09")
    },

    {'n_digits': 7,
     'load_path': weight_path.format("exp_idx=3_repeat=0_seed=63253879_2018_07_12_11_36_55")
    },

    {'n_digits': 9,
     'load_path': weight_path.format("exp_idx=4_repeat=0_seed=1140526616_2018_07_12_11_36_55")
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
    render_hook=yolo_air.YoloAir_ComparisonRenderHook(show_zero_boxes=False),
)


envs.run_experiment(
    "yolo_air_plot", config, readme, distributions=distributions,
    alg="yolo_air", task="arithmetic", durations=durations,
)
