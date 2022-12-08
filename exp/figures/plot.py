import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from figure.util import set_style, set_size, WIDTH, COLOR_LIST


# EXP 3.1
def vit_a100_exp3_multitask_latency():
    labels = ["MIG 40g/40g", "w/ MPS", "w/o MPS"]

    cv_train = [142.36, 108.22, 108.57]
    cv_infer = [43.06, 28.59, 37.33]

    index = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    bar_width = 0.4
    ax.bar(index - bar_width / 2, cv_train, label='Training', color=COLOR_LIST[3], width=bar_width)
    ax.bar(index + bar_width / 2, cv_infer, label='Inference', color=COLOR_LIST[5], width=bar_width)
    ax.set_xticks(index, labels=labels)

    plt.margins(x=0.08)
    ax.set_ylabel('Latency (ms)', fontsize=22)
    ax.set_ylim([0, 170])

    ax.tick_params(axis='y', rotation=90)
    ax.legend(loc='upper right', fontsize=20)

    plt.savefig(f'./exp3_p99_latency_mig_mps_compare.pdf', format='pdf', bbox_inches='tight')


# EXP 3.2
def vit_a100_exp3_multitask_throughput():
    labels = ["MIG 40g/40g", "w/ MPS", "w/o MPS"]
    cv_train = [224.78, 295.7, 294.74]
    cv_infer = [743.2, 1123.37, 861.79]

    index = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    bar_width = 0.4
    ax.bar(index - bar_width / 2, cv_train, label='Training', color=COLOR_LIST[3], width=bar_width)
    ax.bar(index + bar_width / 2, cv_infer, label='Inference', color=COLOR_LIST[5], width=bar_width)
    ax.set_xticks(index, labels=labels)

    plt.margins(x=0.08)
    ax.set_ylabel('Throughput (req/sec)', fontsize=22)
    ax.set_ylim([0, 1300])

    ax.tick_params(axis='y', rotation=90)
    ax.legend(loc='upper left', fontsize=20)

    plt.savefig(f'./exp3_throughput_mig_mps_compare.pdf', format='pdf', bbox_inches='tight')


# EXP 3.3
def vit_a100_exp3_multitask_gract():
    labels = ["MIG 40g/40g", "w/ MPS", "w/o MPS"]
    cv_train = [91, 94, 95]
    cv_infer = [81, 93, 92]

    index = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    bar_width = 0.4
    ax.bar(index - bar_width / 2, cv_train, yerr=np.std(cv_train), label='Training', color=COLOR_LIST[3], width=bar_width)
    ax.bar(index + bar_width / 2, cv_infer, yerr=np.std(cv_infer), label='Inference', color=COLOR_LIST[5], width=bar_width)
    ax.set_xticks(index, labels=labels)

    plt.margins(x=0.08)
    ax.set_ylabel('Graphic Engine Activity (GRACT) (%)', fontsize=18)
    ax.set_ylim([80, 100])
    # ax.set_ylim([0, 180])

    ax.tick_params(axis='y', rotation=90)
    ax.legend(loc='upper left', fontsize=18)

    plt.savefig(f'./exp3_gract_mig_mps_compare.pdf', format='pdf', bbox_inches='tight')


# EXP 3.4
def vit_a100_exp3_multitask_energy():
    labels = ["MIG 40g/40g", "w/ MPS", "w/o MPS"]
    cv_train = [91, 94, 95]
    cv_infer = [81, 93, 92]

    index = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    bar_width = 0.4
    ax.bar(index - bar_width / 2, cv_train, yerr=np.std(cv_train), label='Training', color=COLOR_LIST[3], width=bar_width)
    ax.bar(index + bar_width / 2, cv_infer, yerr=np.std(cv_infer), label='Inference', color=COLOR_LIST[5], width=bar_width)
    ax.set_xticks(index, labels=labels)

    plt.margins(x=0.08)
    ax.set_ylabel('Graphic Engine Activity (GRACT) (%)', fontsize=18)
    ax.set_ylim([80, 100])
    # ax.set_ylim([0, 180])

    ax.tick_params(axis='y', rotation=90)
    ax.legend(loc='upper left', fontsize=18)

    plt.savefig(f'./exp3_gract_mig_mps_compare.pdf', format='pdf', bbox_inches='tight')

# EXP 4.1
def bert_a100_exp3_multitask_latency():
    labels = ["MIG 40g/40g", "w/ MPS", "w/o MPS"]
    bert_train = [142.36, 108.22, 108.57]
    bert_infer = [43.06, 28.59, 37.33]

    index = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    bar_width = 0.4
    ax.bar(index - bar_width / 2, bert_train, yerr=[2.06, 12.78, 8.75], label='Training', color=COLOR_LIST[3], width=bar_width)
    ax.bar(index + bar_width / 2, bert_infer, yerr=[0.2, 2.77, 8.21], label='Inference', color=COLOR_LIST[5], width=bar_width)
    ax.set_xticks(index, labels=labels)

    plt.margins(x=0.08)
    ax.set_ylabel('Avg Latency (ms)', fontsize=22)
    ax.set_ylim([0, 160])

    ax.tick_params(axis='y', rotation=90)
    ax.legend(loc='upper right', fontsize=20)

    plt.savefig(f'./exp4_latency_mig_mps_compare.pdf', format='pdf', bbox_inches='tight')


# EXP 4.1
def bert_a100_exp3_multitask_throughput():
    labels = ["MIG 40g/40g", "w/ MPS", "w/o MPS"]
    bert_train = [48.71, 43.26, 51.26]
    bert_infer = [1251.26, 2266.03, 1608.4]

    index = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    bar_width = 0.4
    ax.bar(index - bar_width / 2, bert_train, label='Training', color=COLOR_LIST[3], width=bar_width)
    ax.bar(index + bar_width / 2, bert_infer, label='Inference', color=COLOR_LIST[5], width=bar_width)
    ax.set_xticks(index, labels=labels)

    plt.yscale('symlog')

    plt.margins(x=0.08)
    ax.set_ylabel('Throughput (sample/sec)', fontsize=22)
    ax.set_ylim([0, 25000])

    ax.tick_params(axis='y', rotation=90)
    ax.legend(loc='upper left', fontsize=20)

    plt.savefig(f'./exp4_throughput_mig_mps_compare.pdf', format='pdf', bbox_inches='tight')


# EXP 4.2
def bert_a100_exp3_multitask_gract():
    labels = ["MIG 40g/40g", "w/ MPS", "w/o MPS"]
    bert_train = [86, 96, 96]

    index = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    bar_width = 0.7
    ax.bar(index, bert_train, label='Training', color=COLOR_LIST[2], width=bar_width)
    ax.set_xticks(index, labels=labels)

    # plt.yscale('symlog')

    plt.margins(x=0.08)
    ax.set_ylabel('Graphic Engine Activity (GRACT) (%)', fontsize=18)
    ax.set_ylim([50, 100])

    ax.tick_params(axis='y', rotation=90)
    # ax.legend(loc='upper left', fontsize=20)

    plt.savefig(f'./exp3_gract_mig_mps_compare.pdf', format='pdf', bbox_inches='tight')

# EXP 4.3
def bert_a100_exp3_multitask_fb():
    labels = ["MIG 40g/40g", "w/ MPS", "w/o MPS"]
    bert_train = [6859, 7399, 7391]

    index = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    bar_width = 0.7
    ax.bar(index, bert_train, label='Training', color=COLOR_LIST[0], width=bar_width)
    ax.set_xticks(index, labels=labels)

    # plt.yscale('symlog')

    plt.margins(x=0.08)
    ax.set_ylabel('Frame Buffer (Mb)', fontsize=20)
    ax.set_ylim([10000, 11000])

    ax.tick_params(axis='y', rotation=90)
    # ax.legend(loc='upper left', fontsize=20)

    plt.savefig(f'./exp4_fb_mig_mps_compare.pdf', format='pdf', bbox_inches='tight')

# EXP 4.4
def bert_a100_exp3_multitask_power():
    labels = ["MIG 40g/40g", "w/ MPS", "w/o MPS"]
    bert_train = [351, 398, 384]

    index = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    bar_width = 0.7
    ax.bar(index, bert_train, label='Training', color=COLOR_LIST[4], width=bar_width)
    ax.set_xticks(index, labels=labels)

    # plt.yscale('symlog')

    plt.margins(x=0.08)
    ax.set_ylabel('Power (Watt)', fontsize=20)
    ax.set_ylim([300, 410])

    ax.tick_params(axis='y', rotation=90)
    # ax.legend(loc='upper left', fontsize=20)

    plt.savefig(f'./exp3_power_mig_mps_compare.pdf', format='pdf', bbox_inches='tight')


# EXP 1.1
def bert_train_seqlen_profile_qps():
    labels = [32, 64, 128, 256, 512]
    data_1g_10gb = [21.16, 15.28, 9.49, 6.43, 0]
    data_2g_20gb = [40.24, 29.33, 18.63, 12.99, 12.13]
    data_3g_40gb = [66.42, 49.45, 30.58, 21.15, 19.84]
    data_7g_80gb = [89.14, 81, 57.44, 40.54, 38.05]

    index = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    line_width = 3
    ax.plot(index, data_1g_10gb, color=COLOR_LIST[0], ms=10, linewidth=line_width, marker=".", label='1g.10gb')
    ax.plot(index, data_2g_20gb, color=COLOR_LIST[1], ms=10, linewidth=line_width, marker=".", label='2g.20gb')
    ax.plot(index, data_3g_40gb, color=COLOR_LIST[2], ms=10, linewidth=line_width, marker=".", label='3g.40gb')
    ax.plot(index, data_7g_80gb, color=COLOR_LIST[3], ms=10, linewidth=line_width, marker=".", label='7g.80gb')

    ax.set_xticks(index, labels=labels)

    # plt.yscale('symlog')

    plt.margins(x=0.08)
    ax.set_ylabel('Throughput (sample/sec)', fontsize=20)
    ax.set_xlabel('Sequence Length', fontsize=20)
    ax.set_ylim([0, 100])

    ax.tick_params(axis='y', rotation=90)
    ax.legend(loc='upper right', fontsize=20)

    plt.savefig(f'./exp1_qps_bert_seq_profile.pdf', format='pdf', bbox_inches='tight')


# EXP 1.1
def bert_train_bs_profile_qps():
    labels = [8, 16, 32, 64, 128, 256]
    data_1g_10gb = [25.85, 20.95, 15.28, 9.92, 5.87, 0]
    data_2g_20gb = [47, 40.17, 29.32, 19.51, 11.77, 6.58]
    data_3g_40gb = [76.13, 67.65, 49.45, 32, 18.85, 10.62]
    data_7g_80gb = [99.29, 90.5, 81.87, 60.53, 37.39, 21.36]

    index = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    line_width = 3
    ax.plot(index, data_1g_10gb, color=COLOR_LIST[0], ms=10, linewidth=line_width, marker=".", label='1g.10gb')
    ax.plot(index, data_2g_20gb, color=COLOR_LIST[1], ms=10, linewidth=line_width, marker=".", label='2g.20gb')
    ax.plot(index, data_3g_40gb, color=COLOR_LIST[2], ms=10, linewidth=line_width, marker=".", label='3g.40gb')
    ax.plot(index, data_7g_80gb, color=COLOR_LIST[3], ms=10, linewidth=line_width, marker=".", label='7g.80gb')

    ax.set_xticks(index, labels=labels)

    # plt.yscale('symlog')

    plt.margins(x=0.08)
    ax.set_ylabel('Throughput (sample/sec)', fontsize=20)
    ax.set_xlabel('Batch Size', fontsize=20)
    ax.set_ylim([0, 110])

    ax.tick_params(axis='y', rotation=90)
    ax.legend(loc='upper right', fontsize=20)

    plt.savefig(f'./exp1_qps_bert_bs_profile.pdf', format='pdf', bbox_inches='tight')


# EXP 1.2 (A30)
def bert_train_bs_profile_latency():
    labels = [1, 2, 4, 8, 16, 32]
    data_1g_6gb = [6.93, 11.97
19.67
38.56
74.1]
    data_2g_12gb = [47, 40.17, 29.32, 19.51, 11.77, 6.58]
    data_4g_24gb = [76.13, 67.65, 49.45, 32, 18.85, 10.62]

    index = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    line_width = 3
    ax.plot(index, data_1g_10gb, color=COLOR_LIST[0], ms=10, linewidth=line_width, marker=".", label='1g.10gb')
    ax.plot(index, data_2g_20gb, color=COLOR_LIST[1], ms=10, linewidth=line_width, marker=".", label='2g.20gb')
    ax.plot(index, data_3g_40gb, color=COLOR_LIST[2], ms=10, linewidth=line_width, marker=".", label='3g.40gb')
    ax.plot(index, data_7g_80gb, color=COLOR_LIST[3], ms=10, linewidth=line_width, marker=".", label='7g.80gb')

    ax.set_xticks(index, labels=labels)

    # plt.yscale('symlog')

    plt.margins(x=0.08)
    ax.set_ylabel('Throughput (sample/sec)', fontsize=20)
    ax.set_xlabel('Batch Size', fontsize=20)
    ax.set_ylim([0, 110])

    ax.tick_params(axis='y', rotation=90)
    ax.legend(loc='upper right', fontsize=20)

    plt.savefig(f'./exp1_qps_bert_bs_profile.pdf', format='pdf', bbox_inches='tight')


# Gateway
if __name__ == "__main__":
    set_style()

    bert_train_seqlen_profile_qps()
    bert_train_bs_profile_qps()
