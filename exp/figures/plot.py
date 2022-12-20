import re
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from util import set_style, set_size, WIDTH, COLOR_LIST

def load_json(file_name, key):
    with open(file_name) as f:
        return dict(json.load(f))[key]


# NEW EXP 1.2 (A30)
def mps_latency():
    labels = [1, 2, 4, 8, 16, 32]
    mps_list, mps_std_list = [], []
    mig_list, mig_std_list = [], []
    
    for bs in labels:
        # mps_std_list.append(1000 * load_json(f'../data/mps/batch_size_2/NVIDIA-A30_resnet18_bs{bs}.json', 'latency_std'))
        # mig_std_list.append(1000 * load_json(f'../data/mig/batch_size_2_2/NVIDIA-A30_resnet18_bs{bs}.json', 'latency_std'))

        # mps_list.append(1000 * load_json(f'../data/mps/batch_size_2/NVIDIA-A30_resnet18_bs{bs}.json', 'latency_mean'))
        # mig_list.append(1000 * load_json(f'../data/mig/batch_size_2_2/NVIDIA-A30_resnet18_bs{bs}.json', 'latency_mean'))

        mps_std_list.append(1000 * load_json(f'../data/mps/batch_size_4/NVIDIA-A30_resnet50_bs{bs}.json', 'latency_std'))
        mig_std_list.append(1000 * load_json(f'../data/mig/batch_size_4_1/NVIDIA-A30_resnet50_bs{bs}.json', 'latency_std'))

        mps_list.append(1000 * load_json(f'../data/mps/batch_size_4/NVIDIA-A30_resnet50_bs{bs}.json', 'latency_mean'))
        mig_list.append(1000 * load_json(f'../data/mig/batch_size_4_1/NVIDIA-A30_resnet50_bs{bs}.json', 'latency_mean'))
        

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

    plt.savefig(f'./set2_a30_mig_mps_resnet50_bs_latency.pdf', format='pdf', bbox_inches='tight')


# NEW EXP 2.1 (A30)
def mps_latency_kde():
    latency_mps = [1000 * x for x in load_json(f'../data/mps/batch_size_4/NVIDIA-A30_resnet50_bs32.json', 'latency')]
    latency_mig = [1000 * x for x in load_json(f'../data/mig/batch_size_4_1/NVIDIA-A30_resnet50_bs32.json', 'latency')]
    fig, ax = plt.subplots(1, 1)

    sns.kdeplot(latency_mps, shade=True, label='MPS (4 processes)', color='green')
    sns.kdeplot(latency_mig, shade=True, label='MIG (4 * 1g.6gb)', color='#005fd4')

    ax.set_xlim([77, 94])
    # ax.set_ylim([0, 0.2])
    ax.set_xlabel('Latency (ms)', fontsize=20)
    ax.set_ylabel('Kernel Density Estimation (KDE)', fontsize=20)
    ax.legend(loc='upper right', fontsize=15)
    # plt.title(f"Latency KDE of XXX", fontsize=10)
    plt.savefig(f'./set2_a30_mig_mps_resnet50_bs_latency_kde.pdf', format='pdf', bbox_inches='tight')


# NEW EXP 2.2 (A30)
def mps_latency_cdf():
    kwargs = {'cumulative': True, 'linewidth': 2.2}

    mig_p95 = load_json(f'../data/mig/batch_size_4_1/NVIDIA-A30_resnet50_bs32.json', 'latency_p95')
    mps_p95 = load_json(f'../data/mps/batch_size_4/NVIDIA-A30_resnet50_bs32.json', 'latency_p95')

    latency_mps = [1000 * x for x in load_json(f'../data/mps/batch_size_4/NVIDIA-A30_resnet50_bs32.json', 'latency')]
    latency_mig = [1000 * x for x in load_json(f'../data/mig/batch_size_4_1/NVIDIA-A30_resnet50_bs32.json', 'latency')]
    fig, ax = plt.subplots(1, 1)

    plt.axvline(mig_p95 * 1000, color='#A52A2A', linestyle = "dashed", linewidth=1.3, ymax=0.95, label=f'P95 Latency')
    plt.axvline(mps_p95 * 1000, color='#A52A2A', linestyle = "dashed", linewidth=1.3, ymax=0.95)
    sns.distplot(latency_mps, hist=False, hist_kws=kwargs, kde_kws=kwargs, label='MPS (4-process)', color='green')
    sns.distplot(latency_mig, hist=False, hist_kws=kwargs, kde_kws=kwargs, label='MIG (4-1g.6gb)', color='#005fd4')

    # ax.set_xlim([0, 250])
    # ax.set_ylim([0, 0.2])
    ax.set_xlabel('Latency (ms)', fontsize=20)
    ax.set_ylabel('CDF', fontsize=20)
    ax.legend(loc='upper left', fontsize=15)
    # plt.title(f"Latency KDE of XXX", fontsize=10)
    plt.savefig(f'./set2_a30_mig_mps_resnet50_bs_latency_cdf.pdf', format='pdf', bbox_inches='tight')


# NEW EXP 2.3 (A30)
def mps_bs_tail_latency():
    labels = [1, 2, 4, 8, 16, 32]
    mig_list, mps_list = [], []

    for bs in labels: 
        mps_list.append(1000 * load_json(f'../data/mps/batch_size_4/NVIDIA-A30_resnet50_bs{bs}.json', 'latency_p99'))
        mig_list.append(1000 * load_json(f'../data/mig/batch_size_4_1/NVIDIA-A30_resnet50_bs{bs}.json', 'latency_p99'))

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

    plt.savefig(f'./exp2_mps_resnet50_bs_a30_4ci.pdf', format='pdf', bbox_inches='tight')


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


# NEW EXP 2.4 (A30)
def mps_latency_cdf_rate():
    client_id = 2
    rate = '200.0'
    kwargs = {'cumulative': True, 'linewidth': 2.2}

    mig_p95 = load_json(f'../data/rate/mig_1x4/NVIDIA-A30_resnet50_bs1_rate{rate}_client{client_id}.json', 'latency_p95')
    mps_p95 = load_json(f'../data/rate/mps_4/NVIDIA-A30_resnet50_bs1_rate{rate}_client{client_id}.json', 'latency_p95')

    latency_mig = [1000 * x for x in load_json(f'../data/rate/mig_1x4/NVIDIA-A30_resnet50_bs1_rate{rate}_client{client_id}.json', 'latency')]
    latency_mps = [1000 * x for x in load_json(f'../data/rate/mps_4/NVIDIA-A30_resnet50_bs1_rate{rate}_client{client_id}.json', 'latency')]

    fig, ax = plt.subplots(1, 1)
    for i in range(4):
        # sns.distplot([1000 * x for x in load_json(f'../data/rate/mps_4/NVIDIA-A30_resnet50_bs1_rate{rate}_client{i}.json', 'latency')], hist=False, hist_kws=kwargs, kde_kws=kwargs, label=f'MPS client-{i}', color=COLOR_LIST[i])
        sns.distplot([1000 * x for x in load_json(f'../data/rate/mig_1x4/NVIDIA-A30_resnet50_bs1_rate{rate}_client{i}.json', 'latency')], hist=False, hist_kws=kwargs, kde_kws=kwargs, label=f'MIG Instance-{i}', color=COLOR_LIST[i])

    # plt.axvline(mig_p95 * 1000, color='#A52A2A', linestyle = "dashed", linewidth=1.3, ymax=0.95, label=f'P95 Latency')
    # plt.axvline(mps_p95 * 1000, color='#A52A2A', linestyle = "dashed", linewidth=1.3, ymax=0.95)

    # sns.distplot(latency_mps, hist=False, hist_kws=kwargs, kde_kws=kwargs, label='MPS (4 processes)', color='green')
    # sns.distplot(latency_mig, hist=False, hist_kws=kwargs, kde_kws=kwargs, label='MIG (4 * 1g.6gb)', color='#005fd4')

    # ax.set_xlim([0, 250])
    # ax.set_ylim([0, 0.2])
    ax.set_xlabel('Latency (ms)', fontsize=20)
    ax.set_ylabel('CDF', fontsize=20)
    ax.legend(loc='lower right', fontsize=15)
    # plt.title(f"Latency KDE of XXX", fontsize=10)
    plt.savefig(f'./set2_a30_mig_mps_resnet50_bs1_latency_rate{rate}_client{client_id}_cdf.pdf', format='pdf', bbox_inches='tight')


# SET 1.1
def set1_resnet50_inference_energy_a100():
    labels = ['1g.10gb', '2g.20gb', '3g.40gb', '4g.40gb', '7g.80gb']
    bs1l, bs2l, bs4l, bs8l, bs16l, bs32l, bs64l = [], [], [], [], [], [], []
    for pr in labels:
        energy_bs1 = load_json(f'../data/gpu_perf/{pr}/NVIDIA-A100-SXM4-80GB_resnet50_bs1.json', 'metrics')['DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION_total']
        energy_bs2 = load_json(f'../data/gpu_perf/{pr}/NVIDIA-A100-SXM4-80GB_resnet50_bs2.json', 'metrics')['DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION_total']
        energy_bs4 = load_json(f'../data/gpu_perf/{pr}/NVIDIA-A100-SXM4-80GB_resnet50_bs4.json', 'metrics')['DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION_total']
        energy_bs8 = load_json(f'../data/gpu_perf/{pr}/NVIDIA-A100-SXM4-80GB_resnet50_bs8.json', 'metrics')['DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION_total']
        energy_bs16 = load_json(f'../data/gpu_perf/{pr}/NVIDIA-A100-SXM4-80GB_resnet50_bs16.json', 'metrics')['DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION_total']
        energy_bs32 = load_json(f'../data/gpu_perf/{pr}/NVIDIA-A100-SXM4-80GB_resnet50_bs32.json', 'metrics')['DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION_total']
        energy_bs64 = load_json(f'../data/gpu_perf/{pr}/NVIDIA-A100-SXM4-80GB_resnet50_bs64.json', 'metrics')['DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION_total']

        bs1l.append(energy_bs1[-1] - energy_bs1[0])
        bs2l.append(energy_bs2[-1] - energy_bs2[0])
        bs4l.append(energy_bs4[-1] - energy_bs4[0])
        bs8l.append(energy_bs8[-1] - energy_bs8[0])
        bs16l.append(energy_bs16[-1] - energy_bs16[0])
        bs32l.append(energy_bs32[-1] - energy_bs32[0])
        bs64l.append(energy_bs64[-1] - energy_bs64[0])

    index = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    line_width = 2.2
    ax.plot(index, bs1l, label='bs=1', color=COLOR_LIST[0], linewidth=line_width)
    ax.plot(index, bs2l, label='bs=2', color=COLOR_LIST[1], linewidth=line_width)
    ax.plot(index, bs4l, label='bs=4', color=COLOR_LIST[2], linewidth=line_width)
    ax.plot(index, bs8l, label='bs=8', color=COLOR_LIST[3], linewidth=line_width)
    ax.plot(index, bs16l, label='bs=16', color=COLOR_LIST[4], linewidth=line_width)
    ax.plot(index, bs32l, label='bs=32', color=COLOR_LIST[5], linewidth=line_width)
    ax.plot(index, bs64l, label='bs=64', color=COLOR_LIST[6], linewidth=line_width)

    ax.set_xticks(index, labels=labels)

    # plt.yscale('log')
    plt.margins(x=0.08)
    ax.set_ylabel('Avg. Energy Consumption (mJ)', fontsize=20)

    ax.tick_params(axis='y', rotation=90)
    ax.legend(loc='upper right', fontsize=15, ncol=2)

    plt.savefig(f'./exp1_resnet50_bs_profile_energy.pdf', format='pdf', bbox_inches='tight')


# Set 1.1
def set1_resnet50_inference_fb_a100():
    labels = ['1g.10gb', '2g.20gb', '3g.40gb', '4g.40gb', '7g.80gb']
    bs1l, bs2l, bs4l, bs8l, bs16l, bs32l, bs64l = [], [], [], [], [], [], []
    for pr in labels:
        bs1l.append(np.mean(load_json(f'../data/gpu_perf/{pr}/NVIDIA-A100-SXM4-80GB_resnet50_bs1.json', 'metrics')['DCGM_FI_DEV_FB_USED']))
        bs2l.append(np.mean(load_json(f'../data/gpu_perf/{pr}/NVIDIA-A100-SXM4-80GB_resnet50_bs2.json', 'metrics')['DCGM_FI_DEV_FB_USED']))
        bs4l.append(np.mean(load_json(f'../data/gpu_perf/{pr}/NVIDIA-A100-SXM4-80GB_resnet50_bs4.json', 'metrics')['DCGM_FI_DEV_FB_USED']))
        bs8l.append(np.mean(load_json(f'../data/gpu_perf/{pr}/NVIDIA-A100-SXM4-80GB_resnet50_bs8.json', 'metrics')['DCGM_FI_DEV_FB_USED']))
        bs16l.append(np.mean(load_json(f'../data/gpu_perf/{pr}/NVIDIA-A100-SXM4-80GB_resnet50_bs16.json', 'metrics')['DCGM_FI_DEV_FB_USED']))
        bs32l.append(np.mean(load_json(f'../data/gpu_perf/{pr}/NVIDIA-A100-SXM4-80GB_resnet50_bs32.json', 'metrics')['DCGM_FI_DEV_FB_USED']))
        bs64l.append(np.mean(load_json(f'../data/gpu_perf/{pr}/NVIDIA-A100-SXM4-80GB_resnet50_bs64.json', 'metrics')['DCGM_FI_DEV_FB_USED']))

    index = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    line_width = 2.2
    ax.plot(index, bs1l, label='bs=1', color=COLOR_LIST[0], linewidth=line_width)
    ax.plot(index, bs2l, label='bs=2', color=COLOR_LIST[1], linewidth=line_width)
    ax.plot(index, bs4l, label='bs=4', color=COLOR_LIST[2], linewidth=line_width)
    ax.plot(index, bs8l, label='bs=8', color=COLOR_LIST[3], linewidth=line_width)
    ax.plot(index, bs16l, label='bs=16', color=COLOR_LIST[4], linewidth=line_width)
    ax.plot(index, bs32l, label='bs=32', color=COLOR_LIST[5], linewidth=line_width)
    ax.plot(index, bs64l, label='bs=64', color=COLOR_LIST[6], linewidth=line_width)

    ax.set_xticks(index, labels=labels)

    plt.margins(x=0.08)
    ax.set_ylim([1000, 10000])
    ax.set_ylabel('Framebuffer Memory Used (MiB)', fontsize=20)

    ax.tick_params(axis='y', rotation=90)
    ax.legend(loc='upper right', fontsize=16, ncol=3)

    plt.savefig(f'./exp1_resnet50_bs_profile_fb.pdf', format='pdf', bbox_inches='tight')

# Set 1.1
def set1_resnet50_inference_gract_a100():
    batch_size = 4
    l1 = load_json(f'../data/gpu_perf/1g.10gb/NVIDIA-A100-SXM4-80GB_resnet50_bs{batch_size}.json', 'metrics')['DCGM_FI_PROF_GR_ENGINE_ACTIVE']
    l2 = load_json(f'../data/gpu_perf/2g.20gb/NVIDIA-A100-SXM4-80GB_resnet50_bs{batch_size}.json', 'metrics')['DCGM_FI_PROF_GR_ENGINE_ACTIVE']
    l3 = load_json(f'../data/gpu_perf/3g.40gb/NVIDIA-A100-SXM4-80GB_resnet50_bs{batch_size}.json', 'metrics')['DCGM_FI_PROF_GR_ENGINE_ACTIVE']
    l4 = load_json(f'../data/gpu_perf/4g.40gb/NVIDIA-A100-SXM4-80GB_resnet50_bs{batch_size}.json', 'metrics')['DCGM_FI_PROF_GR_ENGINE_ACTIVE']
    l7 = load_json(f'../data/gpu_perf/7g.80gb/NVIDIA-A100-SXM4-80GB_resnet50_bs{batch_size}.json', 'metrics')['DCGM_FI_PROF_GR_ENGINE_ACTIVE']
    lno = load_json(f'../data/gpu_perf/no_mig/NVIDIA-A100-SXM4-80GB_resnet50_bs{batch_size}.json', 'metrics')['DCGM_FI_PROF_GR_ENGINE_ACTIVE']

    ls = [l1, l2, l3, l4, l7, lno]

    max_len = 0
    for l in ls:
        if len(l) > max_len:
            max_len = len(l)
    
    for l in ls:
        for _ in range(max_len - len(l) + 1):
            l.append(0)

    index = np.arange(max_len + 1)
    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    line_width = 2.2
    ax.plot(index, l1, label='1g.10gb', color=COLOR_LIST[0], linewidth=line_width)
    ax.plot(index, l2, label='2g.20gb', color=COLOR_LIST[1], linewidth=line_width)
    ax.plot(index, l3, label='3g.40gb', color=COLOR_LIST[2], linewidth=line_width)
    ax.plot(index, l4, label='4g.40gb', color=COLOR_LIST[3], linewidth=line_width)
    ax.plot(index, l7, label='7g.80gb', color=COLOR_LIST[4], linewidth=line_width)
    ax.plot(index, lno, label='w/o MIG', color=COLOR_LIST[5], linewidth=line_width)

    ax.set_xticks(index, labels=index)

    plt.margins(x=0.08)
    ax.set_ylim([0, 1.1])
    # ax.set_xlim([0, 150])
    ax.set_ylabel('GRACT (%)', fontsize=20)

    ax.tick_params(axis='y', rotation=90)
    ax.legend(loc='lower right', fontsize=16, ncol=1)
    ax.xaxis.set_ticks(np.arange(0, 20, 2))
    ax.set_xlabel('Time (sec)', fontsize=20)

    plt.savefig(f'./exp1_resnet50_bs{batch_size}_profile_gract.pdf', format='pdf', bbox_inches='tight')


# Gateway
if __name__ == "__main__":
    set_style()

    # mps_latency()
    # mps_latency_cdf()
    # mps_latency_kde()
    # mps_models_tail_latency()
    # mps_bs_tail_latency()
    # mps_latency_cdf_rate()

    # set1_resnet50_inference_energy_a100()
    # set1_resnet50_inference_fb_a100()
    set1_resnet50_inference_gract_a100()
