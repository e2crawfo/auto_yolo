from auto_yolo import envs
from auto_yolo.models import yolo_air

readme = "Plotting yolo_air."

distributions = None
durations = dict()

config = dict(
    obj_logit_scale=1.0,
    alpha_logit_scale=1.0,
    alpha_logit_bias=1.0,
    obj_temp=1.0,
    training_wheels=0.0,
    render_hook=yolo_air.YoloAir_PaperSetRenderHook(N=32, pred_colour="black", gt_colour="white"),
    render_step=1,
    do_train=False,
    n_train=16,
    load_path="/media/data/Dropbox/experiment_data/active/nips2018/FINAL/for_plot/set/run_search_set_env=task=set_alg=yolo-air_duration=long_seed=0_2018_09_03_21_46_30/experiments/exp_idx=0_repeat=0_seed=899359813_2018_09_03_21_55_57/weights/best_of_stage_0",
    kernel_size=2,
)

envs.run_experiment(
    "plot_set", config, readme, alg="yolo_air",
    task="set", durations=durations, distributions=distributions,
)
