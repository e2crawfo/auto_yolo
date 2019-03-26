from cycler import cycler
import os
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from dps.hyper import HyperSearch
from dps.utils import (
    process_path, Config, sha_cache, set_clear_cache,
    confidence_interval, standard_error
)


data_dir = "/media/data/Dropbox/experiment_data/active/nips2018/FINAL/"
cache_dir = process_path('/home/eric/.cache/dps_plots')
plot_dir = '/media/data/Dropbox/writeups/spatially_invariant_air/figures/'

plot_paths = Config()
plot_paths[''] = ''

verbose_cache = True


def std_dev(ys):
    y_upper = y_lower = [_y.std() for _y in ys]
    return y_upper, y_lower


def ci95(ys):
    conf_int = [confidence_interval(_y, 0.95) for _y in ys]
    y = ys.mean(axis=1)
    y_lower = y - np.array([ci[0] for ci in conf_int])
    y_upper = np.array([ci[1] for ci in conf_int]) - y
    return y_upper, y_lower


def std_err(ys):
    y_upper = y_lower = [standard_error(_y) for _y in ys]
    return y_upper, y_lower


spread_measures = {func.__name__: func for func in [std_dev, ci95, std_err]}


def _get_stage_data_helper(path, stage_idx):
    """
    Return a dataframe, each row of which corresponds to the `stage_idx`-th stage
    of a different run.
    """
    job = HyperSearch(path)
    stage_data = job.extract_stage_data()

    dist_keys = job.dist_keys()

    records = []

    for i, (key, value) in enumerate(sorted(stage_data.items())):
        for (repeat, seed), (df, sc, md) in value.items():
            record = dict(df.iloc[stage_idx])

            for dk in dist_keys:
                record[dk] = md[dk]

            record['idx'] = key.idx
            record['repeat'] = repeat
            record['seed'] = seed

            records.append(record)

    return pd.DataFrame.from_records(records)


def get_stage_data(path, stage_idx, x_key, y_key, spread_measure, y_func=None):
    # Group by idx, get average and spread measure, return x values, mean and spread-measures for y values
    y_func = y_func or (lambda x: x)
    df = _get_stage_data_helper(path, stage_idx)
    groups = sorted(df.groupby(x_key))

    x = [v for v, _df in groups]
    ys = [y_func(_df[y_key]) for v, _df in groups]

    y = [_y.mean() for _y in ys]
    y_upper, y_lower = spread_measures[spread_measure](ys)

    return np.stack([x, y, y_upper, y_lower])


def plot_core_sample_complexity():
    path = "core/run_search_sample_complexity-size=14_colour=False_task=arithmetic_alg=yolo_math_simple_2stage_duration=long_seed=0_2018_05_15_12_59_19"
    path = os.path.join(data_dir, path)

    ax = plt.gca()

    x, y, *yerr = get_stage_data(path, 0, "n_train", "stopping_criteria", "ci95", lambda error: 1 - error)

    label = 'test'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='--')
    plt.show()


@sha_cache(cache_dir, verbose=verbose_cache)
def get_transfer_data(path, x_key, y_key, spread_measure, y_func=None):
    y_func = y_func or (lambda y: y)

    job = HyperSearch(path)
    stage_data = job.extract_stage_data()

    all_data = []

    for i, (key, value) in enumerate(sorted(stage_data.items())):
        data = []

        for (repeat, seed), (df, sc, md) in value.items():
            data.append(df[y_key][1:])  # First stage is the learning stage.

        data = np.array(data).T
        data = y_func(data)

        x = range(1, 21)
        y = data.mean(axis=1)
        yu, yl = spread_measures[spread_measure](data)

        all_data.append(((x, y, yu, yl), key))

    return all_data


@sha_cache(cache_dir, verbose=verbose_cache)
def get_transfer_baseline_data(path, x_key, y_key, spread_measure, y_func=None):
    y_func = y_func or (lambda y: y)

    job = HyperSearch(path)
    stage_data = job.extract_stage_data()

    x = range(1, 21)
    y = []

    for i, (key, value) in enumerate(sorted(stage_data.items())):
        data = []
        for (repeat, seed), (df, sc, md) in value.items():
            data.append(df[y_key][0])
        data = y_func(np.array(data))
        y.append(data.mean())

    return x, np.array(y)


def plot_transfer(extension):
    yolo_path = os.path.join(
        data_dir, "transfer/run_search_yolo-air-transfer_env=size=14-in-colour=False-task=scatter_alg=yolo-air-transfer_duration=long_seed=0_2018_07_19_14_21_39/")
    baseline_path_ap = os.path.join(
        data_dir, "transfer/run_search_transfer-baseline_env=size=14-in-colour=False-task=scatter_alg=baseline_duration=oak_seed=0_2018_07_20_11_27_19/")
    baseline_path_count_1norm = os.path.join(
        data_dir, "transfer/run_search_transfer-baseline-sc=count-1norm_env=size=14-in-colour=False-task=scatter_alg=baseline_duration=oak_seed=0_2018_07_26_12_15_50/")
    baseline_path_count_error = os.path.join(
        data_dir, "transfer/run_search_transfer-baseline-sc=count-error_env=size=14-in-colour=False-task=scatter_alg=baseline_duration=oak_seed=0_2018_07_26_12_29_21/")

    # -----

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.2))
    ax = axes[0]

    y_func = lambda y: 100 * y
    measure = "_test_AP"

    yolo_data = get_transfer_data(yolo_path, "n_train", measure, "ci95", y_func=y_func)

    for (x, y, *yerr), key in yolo_data:
        label = "Trained on {}--{} digits".format(key.min_chars, key.max_chars)
        ax.errorbar(x, y, yerr=yerr, label=label)

    x, y = get_transfer_baseline_data(baseline_path_ap, "n_train", measure, "ci95", y_func=y_func)
    ax.plot(x, y, label="ConnComp")

    fontsize = None
    labelsize = None

    ax.set_ylabel('Average Precision', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.set_ylim((0., 105.))
    ax.set_xlabel('\# Digits in Test Image', fontsize=fontsize)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.legend(loc="lower left", fontsize=8)

    # -----

    ax = axes[1]

    measure = "_test_count_1norm"

    yolo_data = get_transfer_data(yolo_path, "n_train", measure, "ci95")

    for (x, y, *yerr), key in yolo_data:
        label = "Trained on {}--{} digits".format(key.min_chars, key.max_chars)
        ax.errorbar(x, y, yerr=yerr, label=label)

    x, y = get_transfer_baseline_data(baseline_path_count_1norm, "n_train", measure, "ci95")
    ax.plot(x, y, label="ConnComp")

    ax.set_ylabel(r'Count Absolute Error', fontsize=fontsize)
    ax.set_xlabel('\# Digits in Test Image', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.set_ylim((0.0, 3.1))
    ax.set_xticks([0, 5, 10, 15, 20])

    # -----

    ax = axes[2]

    measure = "_test_count_error"

    yolo_data = get_transfer_data(yolo_path, "n_train", measure, "ci95")

    for (x, y, *yerr), key in yolo_data:
        label = "Trained on {}--{} digits".format(key.min_chars, key.max_chars)
        ax.errorbar(x, y, yerr=yerr, label=label)

    x, y = get_transfer_baseline_data(baseline_path_count_error, "n_train", measure, "ci95")
    ax.plot(x, y, label="ConnComp")

    ax.set_ylabel('Count 0-1 Error', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel('\# Digits in Test Image', fontsize=fontsize)
    ax.set_xticks([0, 5, 10, 15, 20])

    plt.subplots_adjust(left=0.07, bottom=0.2, right=0.98, top=0.94, wspace=.27)

    plot_path = os.path.join(plot_dir, 'transfer/main.' + extension)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)
    plt.show()


@sha_cache(cache_dir, verbose=verbose_cache)
def get_arithmetic_data(paths, x_key, y_key, stage_idx, spread_measure, y_func=None):
    y_func = y_func or (lambda y: y)

    data = {}
    for path in paths:
        job = HyperSearch(path)
        stage_data = job.extract_stage_data()

        for i, (key, value) in enumerate(sorted(stage_data.items())):
            _data = []

            for (repeat, seed), (df, sc, md) in value.items():
                _data.append(y_func(df[y_key][stage_idx]))

            data[getattr(key, x_key)] = _data

    x = sorted(data)
    _data = np.array([data[key] for key in x])
    y = _data.mean(axis=1)
    yu, yl = spread_measures[spread_measure](_data)
    return x, y, yu, yl


def plot_addition_old(extension):
    yolo_path = os.path.join(
        data_dir, "addition/2stage/run_search_sample_complexity_experiment_yolo_air_VS_nips_2018_addition_14x14_kind=long_cedar_seed=0_2018_05_14_03_04_29")
    yolo_supplement_path = os.path.join(
        data_dir, "addition/2stage/run_search_supplement_sample_complexity_experiment_yolo_air_VS_nips_2018_addition_14x14_kind=supplement_seed=0_2018_05_14_14_18_26")
    simple_path = os.path.join(
        data_dir, "addition/simple/run_search_sample_complexity-size=14_colour=False_task=addition_alg=yolo_math_simple_duration=long_seed=0_2018_05_14_23_59_50")
    simple_2stage_path = os.path.join(
        data_dir, "addition/simple_2stage/run_search_sample_complexity-size=14_colour=False_task=addition_alg=yolo_math_simple_2stage_duration=long_seed=0_2018_05_15_12_55_38")

    # -----

    fig = plt.figure(figsize=(5, 3.5))
    ax = plt.gca()

    measure = "math_accuracy"

    x, y, *yerr = get_arithmetic_data([yolo_path, yolo_supplement_path], "n_train", measure, 1, "ci95")
    label = "SI-AIR"
    ax.errorbar(x, y, yerr=yerr, label=label)

    x, y, *yerr = get_arithmetic_data([simple_path], "n_train", measure, 0, "ci95")
    label = "Conv"
    ax.errorbar(x, y, yerr=yerr, label=label)

    x, y, *yerr = get_arithmetic_data([simple_2stage_path], "n_train", measure, 0, "ci95")
    label = "Conv - 2stage"
    ax.errorbar(x, y, yerr=yerr, label=label)

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('\# Training Samples / 1000', fontsize=12)
    ax.set_title('Addition - Between 1 and 11 numbers', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 1.05))
    ax.set_xticks(x)
    ax.set_xticklabels((np.array(x) / 1000).astype('i'))

    plt.legend(loc="upper left")
    plot_path = os.path.join(plot_dir, 'addition/main.' + extension)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.subplots_adjust(left=0.12, bottom=0.14, right=0.98, top=0.91)
    fig.savefig(plot_path)
    plt.show()


def plot_arithmetic(extension):
    yolo_path = os.path.join(
        data_dir, "arithmetic/2stage/run_search_sample_complexity-size=14_colour=False_task=arithmetic_alg=yolo_math_2stage_duration=long_seed=0_2018_05_15_00_32_28")
    simple_path = os.path.join(
        data_dir, "arithmetic/simple/run_search_sample_complexity-size=14_colour=False_task=arithmetic_alg=yolo_math_simple_duration=long_seed=0_2018_05_15_00_01_16")
    simple_2stage_path = os.path.join(
        data_dir, "arithmetic/simple_2stage/run_search_sample_complexity-size=14_colour=False_task=arithmetic_alg=yolo_math_simple_2stage_duration=long_seed=0_2018_05_15_12_59_19")

    # -----

    fig = plt.figure(figsize=(5, 3.5))
    ax = plt.gca()

    measure = "math_accuracy"

    x, y, *yerr = get_arithmetic_data([yolo_path], "n_train", measure, 1, "ci95")
    label = "SI-AIR"
    ax.errorbar(x, y, yerr=yerr, label=label)

    x, y, *yerr = get_arithmetic_data([simple_path], "n_train", measure, 0, "ci95")
    label = "Conv"
    ax.errorbar(x, y, yerr=yerr, label=label)

    x, y, *yerr = get_arithmetic_data([simple_2stage_path], "n_train", measure, 0, "ci95")
    label = "Conv - 2stage"
    ax.errorbar(x, y, yerr=yerr, label=label)

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('# Training Samples / 1000', fontsize=12)
    ax.set_title('Arithmetic - Between 1 and 11 numbers', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 1.05))
    ax.set_xticks(x)
    ax.set_xticklabels((np.array(x) / 1000).astype('i'))

    plt.legend(loc="upper left")
    plot_path = os.path.join(plot_dir, 'arithmetic/main.' + extension)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.subplots_adjust(left=0.12, bottom=0.14, right=0.98, top=0.91)
    fig.savefig(plot_path)
    plt.show()


def plot_xo_full_decoder_kind(extension):
    attn_yolo_path = os.path.join(
        data_dir, "xo/full/2stage/run_search_yolo-xo-2stage_env=xo_decoder-kind=attn_alg=yolo-xo-2stage_duration=long_seed=0_2018_06_07_12_29_32")
    mlp_yolo_path = os.path.join(
        data_dir, "xo/full/2stage/run_search_yolo-xo-2stage_env=xo_decoder-kind=mlp_alg=yolo-xo-2stage_duration=long_seed=0_2018_06_07_12_30_29")
    seq_yolo_path = os.path.join(
        data_dir, "xo/full/2stage/run_search_yolo-xo-2stage_env=xo_decoder-kind=seq_alg=yolo-xo-2stage_duration=long_seed=0_2018_06_07_12_30_00")

    attn_simple_path = os.path.join(
        data_dir, "xo/full/simple/run_search_conv-xo_env=xo_decoder-kind=attn_alg=simple-xo_duration=long_seed=0_2018_06_08_17_49_36")
    mlp_simple_path = os.path.join(
        data_dir, "xo/full/simple/run_search_conv-xo_env=xo_decoder-kind=mlp_alg=simple-xo_duration=long_seed=0_2018_06_08_17_50_40")
    seq_simple_path = os.path.join(
        data_dir, "xo/full/simple/run_search_conv-xo_env=xo_decoder-kind=seq_alg=simple-xo_duration=long_seed=0_2018_06_08_17_50_20")

    # -----

    plt.figure(figsize=(5, 3.5))
    ax = plt.gca()

    measure = "math_accuracy"

    x, y, *yerr = get_arithmetic_data([attn_yolo_path], "n_train", measure, 1, "ci95")
    line = ax.errorbar(x, y, yerr=yerr, label="attn-yolo")
    attn_colour = line.lines[0].get_c()

    x, y, *yerr = get_arithmetic_data([mlp_yolo_path], "n_train", measure, 1, "ci95")
    line = ax.errorbar(x, y, yerr=yerr, label="mlp-yolo")
    mlp_colour = line.lines[0].get_c()

    x, y, *yerr = get_arithmetic_data([seq_yolo_path], "n_train", measure, 1, "ci95")
    line = ax.errorbar(x, y, yerr=yerr, label="seq-yolo")
    seq_colour = line.lines[0].get_c()

    # -----

    measure = "math_accuracy"

    x, y, *yerr = get_arithmetic_data([attn_simple_path], "n_train", measure, 0, "ci95")
    ax.errorbar(x, y, yerr=yerr, label="attn-simple", c=attn_colour, ls="--")

    x, y, *yerr = get_arithmetic_data([mlp_simple_path], "n_train", measure, 0, "ci95")
    ax.errorbar(x, y, yerr=yerr, label="mlp-simple", c=mlp_colour, ls="--")

    x, y, *yerr = get_arithmetic_data([seq_simple_path], "n_train", measure, 0, "ci95")
    ax.errorbar(x, y, yerr=yerr, label="seq-simple", c=seq_colour, ls="--")

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('\# Training Samples', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 1.05))
    ax.set_xticks(x)

    plt.legend()
    plt.show()


def plot_xo_2stage_decoder_kind(extension):
    attn_yolo_path = os.path.join(
        data_dir, "xo/pretrained/yolo/run_search_yolo-xo-continue_env=xo_decoder-kind=attn_alg=yolo-xo-continue_duration=long_seed=0_2018_06_07_12_12_19")
    mlp_yolo_path = os.path.join(
        data_dir, "xo/pretrained/yolo/run_search_yolo-xo-continue_env=xo_decoder-kind=mlp_alg=yolo-xo-continue_duration=long_seed=0_2018_06_07_12_17_35")
    seq_yolo_path = os.path.join(
        data_dir, "xo/pretrained/yolo/run_search_yolo-xo-continue_env=xo_decoder-kind=seq_alg=yolo-xo-continue_duration=long_seed=0_2018_06_07_12_13_03")

    attn_simple_path = os.path.join(
        data_dir, "xo/full/simple/run_search_conv-xo_env=xo_decoder-kind=attn_alg=simple-xo_duration=long_seed=0_2018_06_08_17_49_36")
    mlp_simple_path = os.path.join(
        data_dir, "xo/full/simple/run_search_conv-xo_env=xo_decoder-kind=mlp_alg=simple-xo_duration=long_seed=0_2018_06_08_17_50_40")
    seq_simple_path = os.path.join(
        data_dir, "xo/full/simple/run_search_conv-xo_env=xo_decoder-kind=seq_alg=simple-xo_duration=long_seed=0_2018_06_08_17_50_20")

    # -----

    plt.figure(figsize=(5, 3.5))
    ax = plt.gca()

    measure = "math_accuracy"

    x, y, *yerr = get_arithmetic_data([attn_yolo_path], "n_train", measure, 0, "ci95")
    line = ax.errorbar(x, y, yerr=yerr, label="attn-yolo")
    attn_colour = line.lines[0].get_c()

    x, y, *yerr = get_arithmetic_data([mlp_yolo_path], "n_train", measure, 0, "ci95")
    line = ax.errorbar(x, y, yerr=yerr, label="mlp-yolo")
    mlp_colour = line.lines[0].get_c()

    x, y, *yerr = get_arithmetic_data([seq_yolo_path], "n_train", measure, 0, "ci95")
    line = ax.errorbar(x, y, yerr=yerr, label="seq-yolo")
    seq_colour = line.lines[0].get_c()

    # -----

    measure = "math_accuracy"

    x, y, *yerr = get_arithmetic_data([attn_simple_path], "n_train", measure, 0, "ci95")
    ax.errorbar(x, y, yerr=yerr, label="attn-simple", c=attn_colour, ls="--")

    x, y, *yerr = get_arithmetic_data([mlp_simple_path], "n_train", measure, 0, "ci95")
    ax.errorbar(x, y, yerr=yerr, label="mlp-simple", c=mlp_colour, ls="--")

    x, y, *yerr = get_arithmetic_data([seq_simple_path], "n_train", measure, 0, "ci95")
    ax.errorbar(x, y, yerr=yerr, label="seq-simple", c=seq_colour, ls="--")

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('\# Training Samples', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 1.05))
    ax.set_xticks(x)

    plt.legend()
    plt.show()


def plot_comparison(extension):
    yolo_air_path = os.path.join(data_dir, "comparison/run_search_yolo-air-run_env=size=14-in-colour=False-task=arithmetic-ops=addition_alg=yolo-air_duration=long_seed=0_2018_07_16_13_46_48/")
    air_path = os.path.join(data_dir, "comparison/run_search_air-run_env=size=14-in-colour=False-task=arithmetic-ops=addition_alg=attend-infer-repeat_duration=long_seed=0_2018_07_24_13_02_34")
    dair_path = os.path.join(data_dir, "comparison/run_search_dair-run_env=size=14-in-colour=False-task=arithmetic-ops=addition_alg=attend-infer-repeat_duration=long_seed=0_2018_07_10_09_22_24")
    baseline_path = os.path.join(data_dir, "comparison/run_search_comparison-baseline_env=size=14-in-colour=False-task=arithmetic-ops=addition_alg=baseline_duration=oak_seed=0_2018_07_20_11_15_24/")

    # -----

    fig = plt.figure(figsize=(5, 3.5))
    ax = plt.gca()

    y_func = lambda y: 100 * y

    x, y, *yerr = get_arithmetic_data([yolo_air_path], "n_digits", "_test_AP", 0, "ci95", y_func=y_func)
    line = ax.errorbar(x, y, yerr=yerr, label="SPAIR", marker="o", ls="-")
    line.lines[0].get_c()

    x, y, *yerr = get_arithmetic_data([air_path], "n_digits", "_test_AP", 0, "ci95", y_func=y_func)
    line = ax.errorbar(x, y, yerr=yerr, label="AIR", marker="^", ls="-.")
    line.lines[0].get_c()

    x, y, *yerr = get_arithmetic_data([dair_path], "n_digits", "AP", 0, "ci95", y_func=y_func)
    line = ax.errorbar(x, y, yerr=yerr, label="DAIR", marker="v", ls="--")
    line.lines[0].get_c()

    x, y, *yerr = get_arithmetic_data([baseline_path], "n_digits", "_test_AP", 0, "ci95", y_func=y_func)
    ax.plot(x, y, label="ConnComp", marker="s", ls=":")

    ax.set_ylabel('Average Precision', fontsize=12)
    ax.set_xlabel('\# Digits in Image', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0., 105.))
    ax.set_xticks(x)

    plt.legend(loc="upper right", handlelength=4)
    plt.subplots_adjust(left=0.12, bottom=0.13, right=0.99, top=0.99)
    plot_path = os.path.join(plot_dir, 'comparison/main.' + extension)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)
    plt.show()


def plot_addition(extension):
    air_fixed_path = os.path.join(data_dir, "addition/stage2/run_search_addition-stage2_env=task=arithmetic2_run-kind=fixed_alg=air-math_duration=long_seed=0_2018_07_29_09_58_58/")
    baseline_fixed_path = os.path.join(data_dir, "addition/stage2/run_search_addition-stage2_env=task=arithmetic2_run-kind=fixed_alg=baseline-math_duration=long_seed=0_2018_07_28_22_01_21/")
    ground_truth_fixed_path = os.path.join(data_dir, "addition/stage2/run_search_addition-stage2_env=task=arithmetic2_run-kind=fixed_alg=ground-truth-math_duration=long_seed=0_2018_07_28_22_02_14/")
    simple_fixed_path = os.path.join(data_dir, "addition/stage2/run_search_addition-stage2_env=task=arithmetic2_run-kind=fixed_alg=simple-math_duration=long_seed=0_2018_07_28_22_03_18/")
    yolo_air_fixed_path = os.path.join(data_dir, "addition/stage2/run_search_addition-stage2_env=task=arithmetic2_run-kind=fixed_alg=yolo-air-math_duration=long_seed=0_2018_07_28_22_04_04/")

    baseline_raw_path = os.path.join(data_dir, "addition/stage2/run_search_addition-stage2_env=task=arithmetic2_run-kind=raw_alg=baseline-math_duration=long_seed=0_2018_07_28_22_00_58/")
    ground_truth_raw_path = os.path.join(data_dir, "addition/stage2/run_search_addition-stage2_env=task=arithmetic2_run-kind=raw_alg=ground-truth-math_duration=long_seed=0_2018_07_28_22_01_56/")
    simple_raw_path = os.path.join(data_dir, "addition/stage2/run_search_addition-stage2_env=task=arithmetic2_run-kind=raw_alg=simple-math_duration=long_seed=0_2018_07_28_22_02_59/")

    air_unfixed_path = os.path.join(data_dir, "addition/stage2/run_search_addition-stage2_env=task=arithmetic2_run-kind=unfixed_alg=air-math_duration=long_seed=0_2018_07_29_09_58_32/")
    baseline_unfixed_path = os.path.join(data_dir, "addition/stage2/run_search_addition-stage2_env=task=arithmetic2_run-kind=unfixed_alg=baseline-math_duration=long_seed=0_2018_07_28_22_01_39/")
    ground_truth_unfixed_path = os.path.join(data_dir, "addition/stage2/run_search_addition-stage2_env=task=arithmetic2_run-kind=unfixed_alg=ground-truth-math_duration=long_seed=0_2018_07_28_22_02_30/")
    simple_unfixed_path = os.path.join(data_dir, "addition/stage2/run_search_addition-stage2_env=task=arithmetic2_run-kind=unfixed_alg=simple-math_duration=long_seed=0_2018_07_28_22_03_41/")
    yolo_air_unfixed_path = os.path.join(data_dir, "addition/stage2/run_search_addition-stage2_env=task=arithmetic2_run-kind=unfixed_alg=yolo-air-math_duration=long_seed=0_2018_07_28_22_04_21/")

    fig = plt.figure(figsize=(5, 4.5))
    ax = plt.gca()

    x, y, *yerr = get_arithmetic_data([baseline_raw_path], "n_train", "_test_math_accuracy", 0, "ci95")
    line = ax.errorbar(x, y, yerr=yerr, label="ConnComp", marker="o", ls="-")

    x, y, *yerr = get_arithmetic_data([ground_truth_raw_path], "n_train", "_test_math_accuracy", 0, "ci95")
    line = ax.errorbar(x, y, yerr=yerr, label="TrueBB", marker="^", ls="-")

    x, y, *yerr = get_arithmetic_data([simple_raw_path], "n_train", "_test_math_accuracy", 0, "ci95")
    line = ax.errorbar(x, y, yerr=yerr, label="ConvNet", marker="v", ls="-")

    # -----

    x, y, *yerr = get_arithmetic_data([yolo_air_fixed_path], "n_train", "_test_math_accuracy", 0, "ci95")
    line = ax.errorbar(x, y, yerr=yerr, label="SPAIR - Fixed", marker="v", ls="-")
    yolo_air_color = line.lines[0].get_c()

    x, y, *yerr = get_arithmetic_data([yolo_air_unfixed_path], "n_train", "_test_math_accuracy", 0, "ci95")
    line = ax.errorbar(x, y, yerr=yerr, label="SPAIR - Unfixed", marker="v", ls="--", color=yolo_air_color)

    # -----

    x, y, *yerr = get_arithmetic_data([air_fixed_path], "n_train", "_test_math_accuracy", 0, "ci95")
    line = ax.errorbar(x, y, yerr=yerr, label="AIR - Fixed", marker="o", ls="-")
    air_color = line.lines[0].get_c()

    x, y, *yerr = get_arithmetic_data([air_unfixed_path], "n_train", "_test_math_accuracy", 0, "ci95")
    line = ax.errorbar(x, y, yerr=yerr, label="AIR - Unfixed", marker="o", ls="--", color=air_color)

    # -----

    # x, y, *yerr = get_arithmetic_data([baseline_fixed_path], "n_train", "_test_math_accuracy", 0, "ci95")
    # line = ax.errorbar(x, y, yerr=yerr, label="ConnComp - Fixed", marker="o", ls="--")
    # baseline_color = line.lines[0].get_c()

    # x, y, *yerr = get_arithmetic_data([ground_truth_fixed_path], "n_train", "_test_math_accuracy", 0, "ci95")
    # line = ax.errorbar(x, y, yerr=yerr, label="Ground Truth - Fixed", marker="o", ls="--")
    # ground_truth_color = line.lines[0].get_c()

    # x, y, *yerr = get_arithmetic_data([simple_fixed_path], "n_train", "_test_math_accuracy", 0, "ci95")
    # line = ax.errorbar(x, y, yerr=yerr, label="CNN - Fixed", marker="^", ls="--")
    # simple_color = line.lines[0].get_c()

    # -----

    # x, y, *yerr = get_arithmetic_data([baseline_unfixed_path], "n_train", "_test_math_accuracy", 0, "ci95")
    # line = ax.errorbar(x, y, yerr=yerr, label="ConnComp - Unfixed", marker="o", ls="-.", color=baseline_color)

    # x, y, *yerr = get_arithmetic_data([ground_truth_unfixed_path], "n_train", "_test_math_accuracy", 0, "ci95")
    # line = ax.errorbar(x, y, yerr=yerr, label="Ground Truth - Unfixed", marker="o", ls="-.", color=ground_truth_color)

    # x, y, *yerr = get_arithmetic_data([simple_unfixed_path], "n_train", "_test_math_accuracy", 0, "ci95")
    # line = ax.errorbar(x, y, yerr=yerr, label="CNN - Unfixed", marker="^", ls="-.", color=simple_color)

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('\# Training Examples / 1000')
    ax.tick_params(axis='both')
    ax.set_ylim((0., 1.05))
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels((np.array(x) / 1000).astype('i'))

    plt.legend(loc="lower center", handlelength=2.5, bbox_to_anchor=(0.5, 1.01), ncol=3, columnspacing=1)
    plt.subplots_adjust(left=0.10, bottom=0.10, right=0.99, top=0.82)
    plot_path = os.path.join(plot_dir, 'addition/main.' + extension)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)
    plt.show()


def plot_ablation(extension):
    yolo_path_n_lookback_0 = os.path.join(
        data_dir, "ablation/run_search_yolo-air-ablation-n-lookback=0_env=task=scatter_alg=yolo-air-transfer_duration=long_seed=0_2018_09_03_08_52_37/")
    yolo_path_n_lookback_1 = os.path.join(
        data_dir, "ablation/run_search_yolo-air-ablation-n-lookback=1_env=task=scatter_alg=yolo-air-transfer_duration=long_seed=0_2018_09_03_09_00_25/")

    # -----

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.2))
    ax = axes[0]

    y_func = lambda y: 100 * y
    measure = "_test_AP"
    key_filter = lambda key: key.min_chars not in [1, 6]

    nb0_data = get_transfer_data(yolo_path_n_lookback_0, "n_train", measure, "ci95", y_func=y_func)
    for (x, y, *yerr), key in nb0_data:
        if not key_filter(key):
            continue

        label = "No Lateral Conns, {}--{} digits".format(key.min_chars, key.max_chars)
        ax.errorbar(x, y, yerr=yerr, label=label)

    nb1_data = get_transfer_data(yolo_path_n_lookback_1, "n_train", measure, "ci95", y_func=y_func)
    for (x, y, *yerr), key in nb1_data:
        if not key_filter(key):
            continue
        label = "With Lateral Conns, {}--{} digits".format(key.min_chars, key.max_chars)
        ax.errorbar(x, y, yerr=yerr, label=label)

    fontsize = None
    labelsize = None

    ax.set_ylabel('Average Precision', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.set_ylim((0., 105.))
    ax.set_xlabel('\# Digits in Test Image', fontsize=fontsize)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.legend(loc="lower left", fontsize=8)

    # -----

    ax = axes[1]

    measure = "_test_count_1norm"

    nb0_data = get_transfer_data(yolo_path_n_lookback_0, "n_train", measure, "ci95")
    for (x, y, *yerr), key in nb0_data:
        if not key_filter(key):
            continue
        ax.errorbar(x, y, yerr=yerr)

    nb1_data = get_transfer_data(yolo_path_n_lookback_1, "n_train", measure, "ci95")
    for (x, y, *yerr), key in nb1_data:
        if not key_filter(key):
            continue
        ax.errorbar(x, y, yerr=yerr)

    ax.set_ylabel(r'Count Absolute Error', fontsize=fontsize)
    ax.set_xlabel('\# Digits in Test Image', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.set_ylim((0.0, 4.))
    ax.set_xticks([0, 5, 10, 15, 20])

    # -----

    ax = axes[2]

    measure = "_test_count_error"

    nb0_data = get_transfer_data(yolo_path_n_lookback_0, "n_train", measure, "ci95")
    for (x, y, *yerr), key in nb0_data:
        if not key_filter(key):
            continue
        ax.errorbar(x, y, yerr=yerr)

    nb1_data = get_transfer_data(yolo_path_n_lookback_1, "n_train", measure, "ci95")
    for (x, y, *yerr), key in nb1_data:
        if not key_filter(key):
            continue
        ax.errorbar(x, y, yerr=yerr)

    ax.set_ylabel('Count 0-1 Error', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel('\# Digits in Test Image', fontsize=fontsize)
    ax.set_xticks([0, 5, 10, 15, 20])

    # -----

    plt.subplots_adjust(left=0.07, bottom=0.2, right=0.98, top=0.94, wspace=.27)

    plot_path = os.path.join(plot_dir, 'ablation/main.' + extension)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)
    plt.show()


if __name__ == "__main__":
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['times']})
    rc('text', usetex=True)
    rc('text.latex', preamble=r'\usepackage{amsmath}')
    rc('errorbar', capsize=3)

    funcs = {k[5:]: v for k, v in vars().items() if k.startswith('plot_')}

    parser = argparse.ArgumentParser()
    parser.add_argument("plots", nargs='+', help=",".join(sorted(funcs)))

    style_list = ['default', 'classic'] + sorted(style for style in plt.style.available if style != 'classic')
    parser.add_argument("--style", default="bmh", choices=style_list)

    parser.add_argument("--no-block", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--ext", default="pdf")
    args = parser.parse_args()
    plt.rc('lines', linewidth=1)

    color_cycle = plt.get_cmap("Dark2").colors
    # color_cycle = plt.get_cmap("Paired").colors
    # color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown']

    os.makedirs(plot_dir, exist_ok=True)

    if args.clear_cache:
        set_clear_cache(True)

    with plt.style.context(args.style):
        plt.rc('axes', prop_cycle=(cycler('color', color_cycle)))

        print(funcs)

        for name, do_plot in funcs.items():
            if name in args.plots:

                fig = do_plot(args.ext)

                if args.show:
                    plt.show(block=not args.no_block)
