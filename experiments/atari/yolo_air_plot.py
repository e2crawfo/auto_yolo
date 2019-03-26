from auto_yolo import envs
from auto_yolo.models import yolo_air
import argparse

readme = "Plotting yolo_air on atari."

distributions = None
durations = dict()

parser = argparse.ArgumentParser()
parser.add_argument("--game")
args, _ = parser.parse_known_args()

config = dict(
    render_hook=yolo_air.YoloAir_AtariRenderHook(N=32, pred_colour="xkcd:azure"),
    render_step=1,
    do_train=False,
    n_train=16,
    load_path="/media/data/dps_data/logs/atari_env=task=atari/exp_alg=yolo-air_seed=1742161367_2018_09_04_16_03_06/weights/best_of_stage_0",
    kernel_size=2,
    game=args.game,
    postprocessing="",
)

envs.run_experiment(
    "plot_yolo_air_atari_game={}".format(args.game), config, readme, alg="yolo_air",
    task="atari", durations=durations, distributions=distributions,
)
