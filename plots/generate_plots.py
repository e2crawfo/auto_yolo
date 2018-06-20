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


data_dir = "/media/data/Dropbox/experiment_data/active/nips2018/CEDAR/"
cache_dir = process_path('/home/eric/.cache/dps_plots')
plot_dir = './plots'

plot_paths = Config()
plot_paths[''] = ''


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


def gen_ablations():
    plt.figure(figsize=(5, 3.5))

    ax = plt.gca()

    ax.set_ylabel('% Test Error', fontsize=12)
    ax.set_xlabel('# Training Examples', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 100.0))
    ax.set_xscale('log', basex=2)

    spread_measure = 'std_err'

    label_order = []

    x, y, *yerr = _extract_rl_data(ablation_paths['full_interface'], spread_measure, lambda r: 100 * r['test_loss'])
    label = 'Full Interface'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)

    x, y, *yerr = _extract_rl_data(ablation_paths['no_modules'], spread_measure, test_01_loss)
    label = 'No Modules'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)

    x, y, *yerr = _extract_rl_data(ablation_paths['no_transformations'], spread_measure, test_01_loss)
    label = 'No Transformations'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)

    x, y, *yerr = _extract_rl_data(ablation_paths['no_classifiers'], spread_measure, test_01_loss)
    label = 'No Classifiers'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)

    legend_handles = {l: h for h, l in zip(*ax.get_legend_handles_labels())}
    ordered_handles = [legend_handles[l] for l in label_order]

    ax.legend(ordered_handles, label_order, loc='best', ncol=1, fontsize=8)
    plt.subplots_adjust(left=0.16, bottom=0.15, right=0.97, top=0.96)
    fig.savefig(os.path.join(plot_dir, 'ablations.pdf'))
    return fig


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
    # Now group by idx, get average and spread measure, return x values, mean y-values, sprea
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


@sha_cache(cache_dir)
def get_transfer_data(path, x_key, y_key, spread_measure, is_baseline, y_func=None):
    job = HyperSearch(path)
    stage_data = job.extract_stage_data()

    all_data = []

    for i, (key, value) in enumerate(sorted(stage_data.items())):
        data = []

        for (repeat, seed), (df, sc, md) in value.items():
            data.append(df[y_key])
        data = np.array(data).T

        x = range(1, 21)
        if not is_baseline:
            data = data[1:]
        y = data.mean(axis=1)
        yu, yl = spread_measures[spread_measure](data)

        all_data.append(((x, y, yu, yl), key))

    return all_data


def plot_core_transfer():
    yolo_path = os.path.join(
        data_dir, "core/run_search_yolo-air-transfer_env=size=14-in-colour=False-task=scatter_alg=yolo-air-transfer_duration=oak_seed=0_2018_06_11_23_28_04")
    baseline_path = os.path.join(
        data_dir, "core/run_search_yolo-baseline-transfer_env=size=14-in-colour=False-task=scatter_alg=yolo-transfer-baseline_duration=oak_seed=0_2018_06_13_08_43_05")

    # -----

    plt.figure(figsize=(5, 3.5))
    ax = plt.gca()

    measure = "mAP"

    yolo_data = get_transfer_data(yolo_path, "n_train", measure, "ci95", is_baseline=False)

    for (x, y, *yerr), key in yolo_data:
        label = str(tuple(key))
        ax.errorbar(x, y, yerr=yerr, label=label)

    baseline_data = get_transfer_data(baseline_path, "n_train", measure, "ci95", is_baseline=True)

    for (x, y, *yerr), key in baseline_data:
        label = str(tuple(key))
        ax.errorbar(x, y, yerr=yerr, label=label)

    ax.set_ylabel('AP@0.5', fontsize=12)  # Its just AP since we aren't doing multiple classes.
    ax.set_xlabel('# Digits in Image', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 1.05))
    ax.set_xticks([0, 5, 10, 15, 20])

    plt.legend()
    plt.show()

    # -----

    plt.figure(figsize=(5, 3.5))
    ax = plt.gca()

    measure = "count_1norm"

    yolo_data = get_transfer_data(yolo_path, "n_train", measure, "ci95", is_baseline=False)

    for (x, y, *yerr), key in yolo_data:
        label = str(tuple(key))
        ax.errorbar(x, y, yerr=yerr, label=label)

    baseline_data = get_transfer_data(baseline_path, "n_train", measure, "ci95", is_baseline=True)

    for (x, y, *yerr), key in baseline_data:
        label = str(tuple(key))
        ax.errorbar(x, y, yerr=yerr, label=label)

    ax.set_ylabel('|true-count - pred-count|', fontsize=12)
    ax.set_xlabel('# Digits in Image', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 5.0))
    ax.set_xticks([0, 5, 10, 15, 20])

    plt.legend()
    plt.show()

    # -----

    plt.figure(figsize=(5, 3.5))
    ax = plt.gca()

    measure = "count_error"

    yolo_data = get_transfer_data(yolo_path, "n_train", measure, "ci95", is_baseline=False)

    for (x, y, *yerr), key in yolo_data:
        label = str(tuple(key))
        ax.errorbar(x, y, yerr=yerr, label=label)

    baseline_data = get_transfer_data(baseline_path, "n_train", measure, "ci95", is_baseline=True)

    for (x, y, *yerr), key in baseline_data:
        label = str(tuple(key))
        ax.errorbar(x, y, yerr=yerr, label=label)

    ax.set_ylabel('int(true-count != pred-count)', fontsize=12)
    ax.set_xlabel('# Digits in Image', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 1.05))
    ax.set_xticks([0, 5, 10, 15, 20])

    plt.legend()
    plt.show()


@sha_cache(cache_dir)
def get_arithmetic_data(paths, x_key, y_key, stage_idx, spread_measure, y_func=None):
    data = {}
    for path in paths:
        job = HyperSearch(path)
        stage_data = job.extract_stage_data()

        for i, (key, value) in enumerate(sorted(stage_data.items())):
            _data = []

            for (repeat, seed), (df, sc, md) in value.items():
                # Get performance on second stage.
                _data.append(df[y_key][stage_idx])

            data[key.n_train] = _data

    x = sorted(data)
    _data = np.array([data[key] for key in x])
    y = _data.mean(axis=1)
    yu, yl = spread_measures[spread_measure](_data)
    return x, y, yu, yl


def plot_addition(extension):
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
    ax.set_xlabel('# Training Samples / 1000', fontsize=12)
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


def plot_xo_full_decoder_kind():
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
    ax.set_xlabel('# Training Samples', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 1.05))
    ax.set_xticks(x)

    plt.legend()
    plt.show()


def plot_xo_2stage_decoder_kind():
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
    ax.set_xlabel('# Training Samples', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 1.05))
    ax.set_xticks(x)

    plt.legend()
    plt.show()


if __name__ == "__main__":
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

    color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown']
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
