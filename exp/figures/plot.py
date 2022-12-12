import re
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from util import set_style, set_size, WIDTH, COLOR_LIST

def load_json(file_name, key):
    with open(file_name) as f:
        return dict(json.load(f))[key]


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
    data_1g_6gb = [6.93, 11.97, 19.67, 38.56, 74.1]
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


# NEW EXP 1.2 (A30)
def mps_latency():
    labels = [1, 2, 4, 8, 16, 32, 64]
    mps_list, mps_std_list = [], []
    mig_list, mig_std_list = [], []
    
    for bs in labels:
        # mps_std_list.append(1000 * load_json(f'../data/mps/batch_size_2/NVIDIA-A30_resnet18_bs{bs}.json', 'latency_std'))
        # mig_std_list.append(1000 * load_json(f'../data/mig/batch_size_2_2/NVIDIA-A30_resnet18_bs{bs}.json', 'latency_std'))

        # mps_list.append(1000 * load_json(f'../data/mps/batch_size_2/NVIDIA-A30_resnet18_bs{bs}.json', 'latency_mean'))
        # mig_list.append(1000 * load_json(f'../data/mig/batch_size_2_2/NVIDIA-A30_resnet18_bs{bs}.json', 'latency_mean'))

        mps_std_list.append(1000 * load_json(f'../data/mps/batch_size_4/NVIDIA-A30_resnet18_bs{bs}.json', 'latency_std'))
        mig_std_list.append(1000 * load_json(f'../data/mig/batch_size_4_1/NVIDIA-A30_resnet18_bs{bs}.json', 'latency_std'))

        mps_list.append(1000 * load_json(f'../data/mps/batch_size_4/NVIDIA-A30_resnet18_bs{bs}.json', 'latency_mean'))
        mig_list.append(1000 * load_json(f'../data/mig/batch_size_4_1/NVIDIA-A30_resnet18_bs{bs}.json', 'latency_mean'))
        

    index = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    line_width = 3
    ax.errorbar(index, mig_list, color=COLOR_LIST[0], yerr=mig_std_list, ms=10, linewidth=line_width, marker=".", label='MIG (2 instances)')
    ax.errorbar(index, mps_list, color=COLOR_LIST[1], yerr=mps_std_list, ms=10, linewidth=line_width, marker=".", label='MPS (2 processes)')

    ax.set_xticks(index, labels=labels)

    plt.margins(x=0.08)
    ax.set_ylabel('Latency (ms)', fontsize=20)
    ax.set_xlabel('Batch Size', fontsize=20)
    # ax.set_ylim([0, 110])

    ax.tick_params(axis='y', rotation=90)
    ax.legend(loc='upper left', fontsize=20)

    plt.savefig(f'./set2_a30_mig_mps_resnet18_bs_latency.pdf', format='pdf', bbox_inches='tight')


# NEW EXP 2.1 (A30)
def mps_latency_kde():
    latency_mps = [1000 * x for x in load_json(f'../data/mps/batch_size_4/NVIDIA-A30_resnet18_bs64.json', 'latency')]
    latency_mig = [1000 * x for x in load_json(f'../data/mig/batch_size_4_1/NVIDIA-A30_resnet18_bs64.json', 'latency')]
    fig, ax = plt.subplots(1, 1)

    sns.kdeplot(latency_mps, shade=True, label='MPS (4 processes)', color='green')
    sns.kdeplot(latency_mig, shade=True, label='MIG (4 * 1g.6gb)', color='#005fd4')

    # ax.set_xlim([0, 250])
    # ax.set_ylim([0, 0.2])
    ax.set_xlabel('Latency (ms)', fontsize=20)
    ax.set_ylabel('Kernel Density Estimation (KDE)', fontsize=20)
    ax.legend(loc='upper right', fontsize=18)
    # plt.title(f"Latency KDE of XXX", fontsize=10)
    plt.savefig(f'./set2_a30_mig_mps_resnet18_bs_latency_kde.pdf', format='pdf', bbox_inches='tight')


# NEW EXP 2.2 (A30)
def mps_latency_cdf():
    kwargs = {'cumulative': True, 'linewidth': 2.0}

    latency_mps = [1000 * x for x in load_json(f'../data/mps/batch_size_4/NVIDIA-A30_resnet18_bs64.json', 'latency')]
    latency_mig = [1000 * x for x in load_json(f'../data/mig/batch_size_4_1/NVIDIA-A30_resnet18_bs64.json', 'latency')]
    fig, ax = plt.subplots(1, 1)

    sns.distplot(latency_mps, hist=False, hist_kws=kwargs, kde_kws=kwargs, label='MPS (4 processes)', color='green')
    sns.distplot(latency_mig, hist=False, hist_kws=kwargs, kde_kws=kwargs, label='MIG (4 * 1g.6gb)', color='#005fd4')

    # ax.set_xlim([0, 250])
    # ax.set_ylim([0, 0.2])
    ax.set_xlabel('Latency (ms)', fontsize=20)
    ax.set_ylabel('CDF', fontsize=20)
    ax.legend(loc='lower right', fontsize=18)
    # plt.title(f"Latency KDE of XXX", fontsize=10)
    plt.savefig(f'./set2_a30_mig_mps_resnet18_bs_latency_cdf.pdf', format='pdf', bbox_inches='tight')


# NEW EXP 2.3 (A30)
def mps_bs_tail_latency():
    labels = [1, 2, 4, 8, 16, 32, 64]
    mig_list, mps_list = [], []

    for bs in labels: 
        mps_list.append(1000 * load_json(f'../data/mps/batch_size_4/NVIDIA-A30_resnet18_bs{bs}.json', 'latency_p99'))
        mig_list.append(1000 * load_json(f'../data/mig/batch_size_4_1/NVIDIA-A30_resnet18_bs{bs}.json', 'latency_p99'))

    index = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    bar_width = 0.4
    ax.bar(index - bar_width / 2, mig_list, label='MIG (4-instance)', color=COLOR_LIST[4], width=bar_width)
    ax.bar(index + bar_width / 2, mps_list, label='MPS (4-process)', color=COLOR_LIST[5], width=bar_width)
    ax.set_xticks(index, labels=labels)

    # plt.yscale('symlog')
    plt.margins(x=0.08)
    ax.set_ylabel('99th Latency Percentile (ms)', fontsize=20)

    ax.tick_params(axis='y', rotation=90)
    ax.legend(loc='upper left', fontsize=20)

    plt.savefig(f'./exp2_mps_resnet18_bs_a30_4ci.pdf', format='pdf', bbox_inches='tight')


# NEW EXP 2.3 (A30)
def mps_models_tail_latency():
    labels = ["ResNet18", "ResNet34", "ResNet50", "ResNet101"]
    mig_list, mps_list = [], []

    mps_list.append(1000 * load_json(f'../data/mps/model_name_4/NVIDIA-A30_resnet18_bs32.json', 'latency_p95'))
    mps_list.append(1000 * load_json(f'../data/mps/model_name_4/NVIDIA-A30_resnet34_bs32.json', 'latency_p95'))
    mps_list.append(1000 * load_json(f'../data/mps/model_name_4/NVIDIA-A30_resnet50_bs32.json', 'latency_p95'))
    mps_list.append(1000 * load_json(f'../data/mps/model_name_4/NVIDIA-A30_resnet101_bs32.json', 'latency_p95'))

    mig_list.append(1000 * load_json(f'../data/mig/model_name_4_1/NVIDIA-A30_resnet18_bs32.json', 'latency_p95'))
    mig_list.append(1000 * load_json(f'../data/mig/model_name_4_1/NVIDIA-A30_resnet34_bs32.json', 'latency_p95'))
    mig_list.append(1000 * load_json(f'../data/mig/model_name_4_1/NVIDIA-A30_resnet50_bs32.json', 'latency_p95'))
    mig_list.append(1000 * load_json(f'../data/mig/model_name_4_1/NVIDIA-A30_resnet101_bs32.json', 'latency_p95'))

    index = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    bar_width = 0.4
    ax.bar(index - bar_width / 2, mig_list, label='MIG (4-instance)', color=COLOR_LIST[4], width=bar_width)
    ax.bar(index + bar_width / 2, mps_list, label='MPS (4-process)', color=COLOR_LIST[5], width=bar_width)
    ax.set_xticks(index, labels=labels)

    # plt.yscale('symlog')
    plt.margins(x=0.08)
    ax.set_ylabel('95th Latency Percentile (ms)', fontsize=20)

    ax.tick_params(axis='y', rotation=90)
    ax.legend(loc='upper left', fontsize=20)

    plt.savefig(f'./exp2_mps_models_bs32_a30_4ci_95latency.pdf', format='pdf', bbox_inches='tight')


# Gateway
if __name__ == "__main__":
    set_style()

    # mps_latency()
    # mps_latency_cdf()
    # mps_latency_kde()
    # mps_models_tail_latency()
    mps_bs_tail_latency()
