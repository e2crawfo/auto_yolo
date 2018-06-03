from dps import cfg
from dps.utils import Config

from auto_yolo.envs import run_experiment


if __name__ == "__main__":
    _config = Config()
    with _config:
        cfg.update_from_command_line()
    run_experiment("local_run", _config, "")
