import argparse

from dps import cfg
from dps.config import DEFAULT_CONFIG
from dps.train import training_loop
from dps.utils import pdb_postmortem
import dps.projects.nips_2018.envs as env_module
import dps.projects.nips_2018.algs as alg_module


def run():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('env', default=None, help="Name (or unique name-prefix) of environment to run.")
    parser.add_argument('alg', nargs='?', default=None,
                        help="Name (or unique name-prefix) of algorithm to run. Optional. "
                             "If not provided, algorithm spec is assumed to be included "
                             "in the environment spec.")
    parser.add_argument('--pdb', action='store_true',
                        help="If supplied, enter post-mortem debugging on error.")
    args, _ = parser.parse_known_args()

    env = args.env
    alg = args.alg

    if args.pdb:
        with pdb_postmortem():
            _run(env, alg)
    else:
        _run(env, alg)


def _run(env_str, alg_str, _config=None, **kwargs):
    env_config = getattr(env_module, "{}_config".format(env_str))
    alg_config = getattr(alg_module, "{}_config".format(alg_str))

    config = DEFAULT_CONFIG.copy()
    config.update(alg_config)
    config.update(env_config)

    config.log_name = "{}_VERSUS_{}".format(alg_config.log_name, env_config.log_name)

    if _config is not None:
        config.update(_config)
    config.update(kwargs)

    with config:
        cfg.update_from_command_line()
        return training_loop()


if __name__ == "__main__":
    run()
