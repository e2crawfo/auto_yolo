import argparse
import pkgutil

from dps import cfg
from dps.config import DEFAULT_CONFIG
from dps.train import training_loop
from dps.utils import pdb_postmortem
import dps.projects.nips_2018 as env_pkg_nips_2018
import dps.projects.nips_2018.config as alg_module


def get_module_specs(*packages):
    specs = {}
    for p in packages:
        update = {
            name: (loader, name, is_pkg)
            for loader, name, is_pkg in pkgutil.iter_modules(p.__path__)
        }

        intersection = specs.keys() & update.keys()
        assert not intersection, \
            "Module name overlaps: {}".format(list(intersection))
        specs.update(update)
    return specs


def get_module_from_spec(spec):
    return spec[0].find_module(spec[1]).load_module(spec[1])


def parse_env_alg(env, alg):
    env_module_specs = get_module_specs(env_pkg_nips_2018)
    env_spec = env_module_specs[env]
    env_module = get_module_from_spec(env_spec)
    env_config = env_module.config

    alg_config_name = "{}_config".format(alg)
    alg_config = getattr(alg_module, alg_config_name)

    return env_config, alg_config


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
    env_config, alg_config = parse_env_alg(env_str, alg_str)

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
