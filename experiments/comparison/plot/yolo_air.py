from auto_yolo import envs
from auto_yolo.models import yolo_air

readme = "Plotting yolo_air."

distributions = None
durations = dict()

config = dict(
    render_hook=yolo_air.YoloAir_PaperSetRenderHook(N=32, pred_colour="xkcd:azure", gt_colour="xkcd:red"),
    render_step=1,
    do_train=False,
    n_train=16,
    load_path="/media/data/Dropbox/experiment_data/active/nips2018/FINAL/for_plot/comparison/yolo_air/exp_idx=4_repeat=5_seed=1689545029_2018_07_17_06_05_47/weights/best_of_stage_0",
    kernel_size=1,
    n_digits=9,
)

envs.run_experiment(
    "plot_yolo_air_comparison", config, readme, alg="yolo_air",
    task="arithmetic", durations=durations, distributions=distributions,
)
