import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.backends.backend_pgf import FigureCanvasPgf

mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)
pd.set_option('display.max_columns', None)
WIDTH = 506.295
MARKER_LIST = ['.', '*', '^', 'p', 'o', 'x']
HATCH_LIST = [None, '\\', 'x', '-', '/', '+']
COLOR_LIST = ['#DC143C', '#8B008B', '#6495ED', '#3CB371', '#FFD700', '#F08080', '#FF8C00', '#008B8B', '#7B68EE']
short_name = {
        'distiluse-base-multilingual-cased-v1': "distil_v1",
        'distiluse-base-multilingual-cased-v2': "distil_v2",
        'paraphrase-multilingual-MiniLM-L12-v2': "MiniLM",
        'paraphrase-multilingual-mpnet-base-v2': "mpnet",
        'bert-base': "bert-base",
        'resnet50': 'resnet50',
        'vision_transformer': 'vision_transformer'
    }


def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.
    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** .5 - 1) / 1.5

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)
    #     fig_dim = (2*fig_width_in, fig_height_in)

    return fig_dim


def set_style():
    plt.style.use('classic')

    nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 19,
        "font.size": 16,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }



def my_plotter(legends, means_list, x_labels, title, y_axis_name, x_axis_name, save_path, legend_pos, barlabel_fmt, std_list=None):
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))
    y_limit = float(max([max(means) for means in means_list]))*1.4
    index = np.arange(len(x_labels), step=1)
    ax.plot()
    for i, legend in enumerate(legends):
        if i < len(means_list):
            if std_list is None:
                bar = ax.bar(index + 0.2 * i - 0.3, means_list[i], color=COLOR_LIST[i], label=str(legend) , width=0.15)
            else:
                bar = ax.bar(index+0.2*i-0.3, means_list[i], yerr=std_list[i], color=COLOR_LIST[i], label=legend, width=0.15)
            ax.bar_label(bar, label_type='edge', fmt=barlabel_fmt, fontsize=4)

    plt.margins(x=0.08)

    ax.set_xlabel(x_axis_name, fontsize=16)
    ax.set_ylabel(y_axis_name, fontsize=16)
    ax.set_ylim([0, y_limit])

    ax.tick_params(axis='x', rotation=0)
    ax.set_xticks(index)
    ax.set_xticklabels(x_labels, fontsize=14)

    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    yticks[1].label1.set_visible(False)

    ax.legend(loc=legend_pos, ncol=2, fontsize=13)
    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, format='svg')
    plt.close()


def model_compare_draw(result: pd.DataFrame, result_dir):
    instances = []
    latency_list = []
    latency_std_list = []
    throughput_list = []
    gract_list = []
    gract_std_list = []
    fbusd_list = []
    fbusd_std_list = []
    result = result.drop(result[(result.batch_size!=result.loc[1, 'batch_size'])&(result.seq_length!=result.loc[1, 'seq_length'])].index)
    for instance, model_instance_group in result.groupby('mig_profile'):
        instances += [str(instance)]
        latency_list += [model_instance_group.loc[:, 'latency']]
        latency_std_list += [model_instance_group.loc[:, 'latency_std']]
        throughput_list += [model_instance_group.loc[:, 'throughput']]
        gract_list += [model_instance_group.loc[:, 'gract']]
        gract_std_list += [model_instance_group.loc[:, 'gract_std']]
        fbusd_list += [model_instance_group.loc[:, 'fbusd']]
    my_plotter(
        title="multilingual models latency comparison",
        legends=instances,
        means_list=latency_list,
        std_list=latency_std_list,
        x_labels=[short_name[model_name]for model_name in model_instance_group.loc[:, 'model_name']],
        x_axis_name="model name",
        y_axis_name="latency(ms)",
        save_path=f"{result_dir}models_compare_latency.svg",
        legend_pos='upper right',
        barlabel_fmt='%.0f',
    )
    my_plotter(
        title="multilingual models throughput comparison",
        legends=instances,
        means_list=throughput_list,
        x_labels=[short_name[model_name] for model_name in model_instance_group.loc[:, 'model_name']],
        x_axis_name="model name",
        y_axis_name="throughput(/s)",
        save_path=f"{result_dir}models_compare_throughput.svg",
        legend_pos='upper right',
        barlabel_fmt='%.0f',
    )
    my_plotter(
        title="multilingual models graphics engine activity comparison",
        legends=instances,
        means_list=gract_list,
        std_list=gract_std_list,
        x_labels=[short_name[model_name] for model_name in model_instance_group.loc[:, 'model_name']],
        x_axis_name="model name",
        y_axis_name="Graphics Engine Activity",
        save_path=f"{result_dir}models_compare_gract.svg",
        legend_pos='upper right',
        barlabel_fmt='%.2f',
    )
    my_plotter(
        title="multilingual models frame buffer used comparison",
        legends=instances,
        means_list=fbusd_list,
        std_list=fbusd_std_list,
        x_labels=[short_name[model_name] for model_name in model_instance_group.loc[:, 'model_name']],
        x_axis_name="model name",
        y_axis_name="FB used",
        save_path=f"{result_dir}models_compare_fbusd.svg",
        legend_pos='upper right',
        barlabel_fmt='%.0f',
    )


def bsz_draw(result: pd.DataFrame, picture_dir):
        for model_name, model_group in result.groupby('model_name'):
            instances = []
            latency_list = []
            latency_std_list = []
            throughput_list = []
            gract_list = []
            gract_std_list = []
            fbusd_list = []
            power_list = []

            for instance, model_instance_group in model_group.groupby('mig_profile'):
                instances += [str(instance)]
                latency_list += [model_instance_group.loc[:, 'latency']]
                latency_std_list += [model_instance_group.loc[:, 'latency_std']]
                throughput_list += [model_instance_group.loc[:, 'throughput']]
                gract_list += [model_instance_group.loc[:, 'gract']]
                gract_std_list += [model_instance_group.loc[:, 'gract_std']]
                fbusd_list += [model_instance_group.loc[:, 'fbusd']]
                power_list += [model_instance_group.loc[:, 'power']]

            my_plotter(
                legends=instances,
                means_list=fbusd_list,
                x_labels=model_instance_group.loc[:, 'batch_size'],
                title=model_name,
                x_axis_name="batch size",
                y_axis_name="FB used",
                save_path=f"{picture_dir}/{short_name[model_name]}_fbusd_bsz_compare.svg",
                legend_pos='upper right',
                barlabel_fmt='%.0f'
            )
            my_plotter(
                legends=instances,
                means_list=gract_list,
                std_list=gract_std_list,
                title=model_name,
                x_labels=model_instance_group.loc[:, 'batch_size'],
                x_axis_name="batch size",
                y_axis_name="Graphics Engine Activity",
                save_path=f"{picture_dir}/{short_name[model_name]}_gract_bsz_compare.svg",
                legend_pos='upper right',
                barlabel_fmt='%.2f'
            )
            my_plotter(
                legends=instances,
                means_list=latency_list,
                std_list=latency_std_list,
                title=model_name,
                x_labels=model_instance_group.loc[:, 'batch_size'],
                x_axis_name="batch size",
                y_axis_name="latency(ms)",
                save_path=f"{picture_dir}/{short_name[model_name]}_latency_bsz_compare.svg",
                legend_pos='upper right',
                barlabel_fmt='%.0f'
            )
            my_plotter(
                legends=instances,
                means_list=throughput_list,
                title=model_name,
                x_labels=model_instance_group.loc[:, 'batch_size'],
                x_axis_name="batch size",
                y_axis_name="throughput(/s)",
                save_path=f"{picture_dir}/{short_name[model_name]}_throughput_bsz_compare.svg",
                legend_pos='upper right',
                barlabel_fmt='%.0f'
            )
            my_plotter(
                legends=instances,
                means_list=power_list,
                title=model_name,
                x_labels=model_instance_group.loc[:, 'batch_size'],
                x_axis_name="batch size",
                y_axis_name="power(watt)",
                save_path=f"{picture_dir}/{short_name[model_name]}_power_bsz_compare.svg",
                legend_pos='upper right',
                barlabel_fmt='%.0f'
            )


def seq_draw(result: pd.DataFrame, picture_dir):
    for model_name, model_group in result.groupby('model_name'):
        instances = []
        latency_list = []
        latency_std_list = []
        throughput_list = []
        gract_list = []
        gract_std_list = []
        fbusd_list = []
        power_list = []
        for instance, model_instance_group in model_group.groupby('mig_profile'):
            instances += [str(instance)]
            latency_list += [model_instance_group.loc[:, 'latency']]
            latency_std_list += [model_instance_group.loc[:, 'latency_std']]
            throughput_list += [model_instance_group.loc[:, 'throughput']]
            gract_list += [model_instance_group.loc[:, 'gract']]
            gract_std_list += [model_instance_group.loc[:, 'gract_std']]
            fbusd_list += [model_instance_group.loc[:, 'fbusd']]
            power_list += [model_instance_group.loc[:, 'power']]

        my_plotter(
            legends=instances,
            means_list=fbusd_list,
            x_labels=model_instance_group.loc[:, 'seq_length'],
            title=model_name,
            x_axis_name="sequence length",
            y_axis_name="FB used",
            save_path=f"{picture_dir}/{short_name[model_name]}_fbusd_seq_compare.svg",
            legend_pos='upper right',
            barlabel_fmt='%.0f'
        )
        my_plotter(
            legends=instances,
            means_list=gract_list,
            std_list=gract_std_list,
            title=model_name,
            x_labels=model_instance_group.loc[:, 'seq_length'],
            x_axis_name="sequence length",
            y_axis_name="Graphics Engine Activity",
            save_path=f"{picture_dir}/{short_name[model_name]}_gract_seq_compare.svg",
            legend_pos='upper right',
            barlabel_fmt='%.2f'
        )
        my_plotter(
            legends=instances,
            means_list=latency_list,
            std_list=latency_std_list,
            title=model_name,
            x_labels=model_instance_group.loc[:, 'seq_length'],
            x_axis_name="sequence length",
            y_axis_name="latency(ms)",
            save_path=f"{picture_dir}/{short_name[model_name]}_latency_seq_compare.svg",
            legend_pos='upper right',
            barlabel_fmt='%.0f'
        )
        my_plotter(
            legends=instances,
            means_list=throughput_list,
            title=model_name,
            x_labels=model_instance_group.loc[:, 'seq_length'],
            x_axis_name="sequence length",
            y_axis_name="throughput(/s)",
            save_path=f"{picture_dir}/{short_name[model_name]}_throughput_seq_compare.svg",
            legend_pos='upper right',
            barlabel_fmt='%.0f'
        )
        my_plotter(
            legends=instances,
            means_list=power_list,
            title=model_name,
            x_labels=model_instance_group.loc[:, 'seq_length'],
            x_axis_name="sequence length",
            y_axis_name="power(watt)",
            save_path=f"{picture_dir}/{short_name[model_name]}_power_seq_compare.svg",
            legend_pos='upper right',
            barlabel_fmt='%.0f'
        )


if __name__ == '__main__':
    result = pd.read_csv('E:\MIGProfiler\data/results\A30/bert_seq_train.txt')
    seq_draw(result, "E:\MIGProfiler\data\pictures\A30/train")
