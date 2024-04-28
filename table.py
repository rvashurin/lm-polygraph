from lm_eval.tasks import get_task_dict
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.ue_metrics.pred_rej_area import PredictionRejectionArea
from lm_polygraph.ue_metrics.ue_metric import (
    get_random_scores,
    normalize_metric,
)
from collections import defaultdict
import json
import pickle
import numpy as np
from texttable import Texttable
import latextable

TOTAL_TASK = "mmlu"
ENTROPY_NAMES = [f"RouterEntropy{i}" for i in range(32)]
UE_NAMES = {
    "MaximumSequenceProbability": "MSP",
    "Perplexity": "PPL",
    "MeanTokenEntropy": "MTE",
    "TotalMeanExpertEntropy": "TMEE",
    "TotalEntropyOfExpertMean": "TEEM",
    "FirstTokenMeanExpertEntropy": "FTMEE",
    "FirstTokenEntropyOfExpertMean": "FTEEM"
}

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
base_mean_accs, base_mean_ems = [], []

ue_metric = PredictionRejectionArea()

accs_tt = Texttable()
accs_tt.add_row(['', 'Base Acc'] + list(UE_NAMES.values()) + ['Best Layer', 'Best Layer Score'])
ems_tt = Texttable()
ems_tt.add_row(['', 'Base EM'] + list(UE_NAMES.values()) + ['Best Layer', 'Best Layer Score'])

accs_rows, ems_rows = [], []

def get_metric_row(estimations, task_name, metrics):
    metrics = np.array(metrics)
    mean_metric = np.mean(metrics)

    scores = get_man_scores(estimations, UE_NAMES.keys(), metrics)
    layerwise_scores = get_man_scores(estimations, ENTROPY_NAMES, metrics)
    best_layerwise_layer = np.argmax(layerwise_scores)
    best_layerwise_score = layerwise_scores[best_layerwise_layer]
    best_method = np.argmax(scores)

    scores[best_method] = f'\\textbf{{{scores[best_method]}}}'

    return mean_metric, [task_name.replace('_', '\_')] + [mean_metric] + scores + [best_layerwise_layer, best_layerwise_score]

def get_man_scores(estimations, ue_names, metrics):
    scores = []

    for ue_name in ue_names:
        ues = np.array(estimations[("sequence", ue_name)])

        prr = get_normalized_prr(ues, metrics)
        scores.append(round(prr, 3))

    return scores

def get_normalized_prr(ues, metrics):
    prr = ue_metric(ues, metrics)
    prr_oracle = ue_metric(-metrics, metrics)
    prr_random = get_random_scores(ue_metric, metrics)

    return normalize_metric(prr, prr_oracle, prr_random)

for base_task in base_tasks:
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

    mean_acc, accs_row = get_metric_row(manager.estimations, base_task.replace('mmlu_', ''), accs)
    mean_em, ems_row = get_metric_row(manager.estimations, base_task.replace('mmlu_', ''), ems)

    base_mean_accs.append(mean_acc)
    base_mean_ems.append(mean_em)

    accs_rows.append(accs_row)
    ems_rows.append(ems_row)

acc_row_order = np.argsort(base_mean_accs)[::-1]
ems_row_order = np.argsort(base_mean_ems)[::-1]

for i in acc_row_order:
    accs_tt.add_row(accs_rows[i])

for i in ems_row_order:
    ems_tt.add_row(ems_rows[i])

accs_rows = []
ems_rows = []
mean_accs = []
mean_ems = []
for group, task_names in group_tasks.items():
    accs = np.concatenate([base_accs[task_name] for task_name in task_names])
    ems = np.concatenate([base_ems[task_name] for task_name in task_names])
    
    ues = {}
    for ue_name in UE_NAMES:
        ues[("sequence", ue_name)] = np.concatenate([base_ues[task][("sequence", ue_name)] for task in task_names])
    for ue_name in ENTROPY_NAMES:
        ues[("sequence", ue_name)] = np.concatenate([base_ues[task][("sequence", ue_name)] for task in task_names])

    mean_acc, accs_row = get_metric_row(ues, group, accs)
    mean_em, ems_row = get_metric_row(ues, group, ems)

    mean_accs.append(mean_acc)
    mean_ems.append(mean_em)

    accs_rows.append(accs_row)
    ems_rows.append(ems_row)

acc_row_order = np.argsort(mean_accs)[::-1]
ems_row_order = np.argsort(mean_ems)[::-1]

for i in acc_row_order:
    accs_tt.add_row(accs_rows[i])

for i in ems_row_order:
    ems_tt.add_row(ems_rows[i])


accs = np.concatenate(list(base_accs.values()))
ems = np.concatenate(list(base_ems.values()))

ues = {}
for ue_name in UE_NAMES:
    ues[("sequence", ue_name)] = np.concatenate([base_ues[task][("sequence", ue_name)] for task in base_tasks])
for ue_name in ENTROPY_NAMES:
    ues[("sequence", ue_name)] = np.concatenate([base_ues[task][("sequence", ue_name)] for task in base_tasks])

_, accs_row = get_metric_row(ues, 'mmlu', accs)
_, ems_row = get_metric_row(ues, 'mmlu', ems)

accs_tt.add_row(accs_row)
ems_tt.add_row(ems_row)

with open(f'accs.tex', 'w') as f:
    f.write(latextable.draw_latex(accs_tt, caption="an example table.", label="table:example_table"))

with open(f'ems.tex', 'w') as f:
    f.write(latextable.draw_latex(ems_tt, caption="an example table.", label="table:example_table"))

