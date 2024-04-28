from lm_eval.tasks import get_task_dict
from lm_polygraph.utils.manager import UEManager
from lm_eval.ue_metrics import PredictionRejectionArea
from collections import defaultdict
from matplotlib import pyplot as plt
import json
import pickle
import numpy as np


TOTAL_TASK = "mmlu"
ENTROPY_NAMES = [f"RouterEntropy{i}" for i in range(33)]
UE_NAMES = [
    "MaximumSequenceProbability",
    "Perplexity",
    "MeanTokenEntropy",
    "TotalMeanExpertEntropy",
    "TotalEntropyOfExpertMean",
    "FirstTokenMeanExpertEntropy",
    "FirstTokenEntropyOfExpertMean",
] + ENTROPY_NAMES

# group_tasks = defaultdict(list)
# base_tasks = []
# task_dict = get_task_dict(['mmlu'])
#
# for task_name, (group, task) in task_dict.items():
#    if task_name != TOTAL_TASK:
#        if task is not None:
#            base_tasks.append(task.config.task)
#        if group != TOTAL_TASK:
#            group_tasks[group].append(task_name)
#
# with open('task_dicts.pickle', 'wb') as f:
#    pickle.dump((group_tasks, base_tasks), f)

rej_rate = np.linspace(0.0, 0.9, 10)

with open("task_dicts.pickle", "rb") as f:
    group_tasks, base_tasks = pickle.load(f)

base_ues, base_accs, base_ems = {}, {}, {}


def reject(rates, ues, metric):
    order = np.argsort(ues)
    sorted_metric = np.array(metric)[order]

    rej_metric = []
    for rate in rates:
        rej_metric.append(sorted_metric[: int((1 - rate) * len(order))].mean())

    return rej_metric


def reject_randomly(rates, metric):
    rej_metric = []
    for rate in rates:
        rej_metric.append(
            np.random.choice(
                metric, int((1 - rate) * len(metric)), replace=False
            ).mean()
        )

    return rej_metric


def reject_by_oracle(rates, metric):
    sorted_metric = np.array(sorted(metric)[::-1])

    rej_metric = []
    for rate in rates:
        rej_metric.append(sorted_metric[: int((1 - rate) * len(metric))].mean())

    return rej_metric


fig, axs = plt.subplots(len(base_tasks), 2, figsize=(12, 6 * len(base_tasks)))

fig = plt.figure(constrained_layout=True, figsize=(12, len(base_tasks) * 6))
fig.suptitle("Mixtral Rejection - MMLU")
subfigs = fig.subfigures(nrows=len(base_tasks), ncols=1)

for subfig, base_task in zip(subfigs, base_tasks):
    subfig.suptitle(f"{base_task}")

    manager = UEManager.load(f"./polygraph_with_layerwise_entropies/{base_task}_ue_manager_seed1")
    harness_res = json.load(
        open(f"./harness/pretrained__mistralai__Mixtral-8x7B-v0.1_{base_task}.jsonl")
    )

    accs, ems = [], []
    for res in harness_res:
        accs.append(res["acc"])
        ems.append(res["exact_match"])
    base_accs[base_task] = accs
    base_ems[base_task] = ems
    base_ues[base_task] = manager.estimations

    rej_accs_random = reject_randomly(rej_rate, accs)
    rej_accs_oracle = reject_by_oracle(rej_rate, accs)

    rej_ems_random = reject_randomly(rej_rate, ems)
    rej_ems_oracle = reject_by_oracle(rej_rate, ems)

    # create 1x3 subplots per subfig
    axs = subfig.subplots(nrows=1, ncols=2)

    axs[0].set_ylabel("Accuracy")
    axs[0].plot(rej_rate, rej_accs_random, label="Random")
    axs[0].plot(rej_rate, rej_accs_oracle, label="Oracle")
    axs[1].set_ylabel("Exact match")
    axs[1].plot(rej_rate, rej_ems_random, label="Random")
    axs[1].plot(rej_rate, rej_ems_oracle, label="Oracle")

    for ue_name in UE_NAMES:
        ues = manager.estimations[("sequence", ue_name)]

        rej_accs = reject(rej_rate, ues, accs)
        rej_ems = reject(rej_rate, ues, ems)
        
        if ue_name in ENTROPY_NAMES:
            axs[0].plot(rej_rate, rej_accs)
            axs[1].plot(rej_rate, rej_ems)
        else:
            axs[0].plot(rej_rate, rej_accs, label=ue_name)
            axs[1].plot(rej_rate, rej_ems, label=ue_name)

    for ax in axs:
        ax.legend()
        ax.set_xlabel("Rej rate")

plt.savefig(f"mixtral_mmlu_tasks.png")
plt.clf()

for group, task_names in group_tasks.items():
    accs = np.concatenate([base_accs[task_name] for task_name in task_names])
    ems = np.concatenate([base_ems[task_name] for task_name in task_names])

    rej_accs_random = reject_randomly(rej_rate, accs)
    rej_accs_oracle = reject_by_oracle(rej_rate, accs)

    rej_ems_random = reject_randomly(rej_rate, ems)
    rej_ems_oracle = reject_by_oracle(rej_rate, ems)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_ylabel("Accuracy")
    axs[0].plot(rej_rate, rej_accs_random, label="Random")
    axs[0].plot(rej_rate, rej_accs_oracle, label="Oracle")
    axs[1].set_ylabel("Exact match")
    axs[1].plot(rej_rate, rej_ems_random, label="Random")
    axs[1].plot(rej_rate, rej_ems_oracle, label="Oracle")

    for ue_name in UE_NAMES:
        ues = np.concatenate(
            [base_ues[task_name][("sequence", ue_name)] for task_name in task_names]
        )

        rej_accs = reject(rej_rate, ues, accs)
        rej_ems = reject(rej_rate, ues, ems)

        if ue_name in ENTROPY_NAMES:
            axs[0].plot(rej_rate, rej_accs)
            axs[1].plot(rej_rate, rej_ems)
        else:
            axs[0].plot(rej_rate, rej_accs, label=ue_name)
            axs[1].plot(rej_rate, rej_ems, label=ue_name)

    for ax in axs:
        ax.legend()
        ax.set_xlabel("Rej rate")

    fig.suptitle(group)
    plt.savefig(f"{group}.png")
    plt.clf()

accs = np.concatenate(list(base_accs.values()))
ems = np.concatenate(list(base_ems.values()))

rej_accs_random = reject_randomly(rej_rate, accs)
rej_accs_oracle = reject_by_oracle(rej_rate, accs)

rej_ems_random = reject_randomly(rej_rate, ems)
rej_ems_oracle = reject_by_oracle(rej_rate, ems)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].set_ylabel("Accuracy")
axs[0].plot(rej_rate, rej_accs_random, label="Random")
axs[0].plot(rej_rate, rej_accs_oracle, label="Oracle")
axs[1].set_ylabel("Exact match")
axs[1].plot(rej_rate, rej_ems_random, label="Random")
axs[1].plot(rej_rate, rej_ems_oracle, label="Oracle")

for ue_name in UE_NAMES:
    ues = np.concatenate(
        [base_ues[task_name][("sequence", ue_name)] for task_name in base_tasks]
    )

    rej_accs = reject(rej_rate, ues, accs)
    rej_ems = reject(rej_rate, ues, ems)

    if ue_name in ENTROPY_NAMES:
        axs[0].plot(rej_rate, rej_accs)
        axs[1].plot(rej_rate, rej_ems)
    else:
        axs[0].plot(rej_rate, rej_accs, label=ue_name)
        axs[1].plot(rej_rate, rej_ems, label=ue_name)

for ax in axs:
    #ax.legend()
    ax.set_xlabel("Rej rate")

fig.suptitle("Total")
plt.savefig(f"total.png")
plt.clf()
