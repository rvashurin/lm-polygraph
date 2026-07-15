"""Phase-0 diagnostic: is UGRIP direct-mode weakness just 0-shot, or deeper?

Internal ablation (no paper reference numbers). For each (model, dataset) it
compares three conditions and reports accuracy + normalized PRR, with the delta
vs the current baseline:

  C0  direct 0-shot        current polygraph_eval_ugrip.yaml on
                           UGRIP-LM-Polygraph/<ds>-direct  (what we run today)
  C1  direct few-shot      same, but K exemplars sampled from the dataset's
                           train split are prepended to every question. Injected
                           without touching core code via a local CSV fed through
                           Dataset.from_csv (the chat template still applies).
  C2  paper config         the paper's own polygraph_eval_<ds>.yaml
                           (LM-Polygraph/<ds> 'continuation', few-shot baked in,
                           generate_until=['\\n'], tiny max_new_tokens) with the
                           model overridden to ours.

MMLU is the clean cross-condition anchor (no-CoT few-shot in all three). The
paper's GSM8k config is chain-of-thought, so GSM8k-C2 is NOT a no-CoT comparison
and is flagged as such. MedMCQA has no paper config (C0/C1 only).

Usage (run the three stages in order; the middle stage runs on GPUs):

  # 1. build the C1 few-shot CSVs (needs the HF datasets; CPU only)
  python scripts/diagnose_fewshot.py build

  # 2. print the polygraph_eval commands for every (condition, model, dataset).
  #    Redirect into a shell/SLURM job and run on the cluster.
  python scripts/diagnose_fewshot.py commands > run_diagnose.sh

  # 3. once the .man files exist, tabulate the comparison
  python scripts/diagnose_fewshot.py analyze

All stages share OUT_DIR (default ./workdir/diagnose_fewshot); each run writes to
OUT_DIR/<condition>__<model>__<dataset>/ so `analyze` can recover the labels.
"""

import argparse
import csv
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration — edit the grid here.
# --------------------------------------------------------------------------- #

OUT_DIR = Path("./workdir/diagnose_fewshot")
N_SHOT = 5
SUBSAMPLE = 200
SEED = 1

# short-name -> model config (HF whitebox, as the direct SLURM uses — needed for
# log-likelihoods / TokenSAR).
MODELS = {
    "llama": "ugrip_llama_instruct",
    "qwen": "ugrip_qwen25_instruct",
}

# per-dataset knobs. `paper_config` = None means "no paper config" (C0/C1 only).
# `c2_extract` = whether to apply our extractor to the paper-config run too, so
# the accuracy axis is comparable (only safe when the paper config sets no
# normalize / ignore_regex — true for mmlu, false for gsm8k which is CoT).
DATASETS = {
    "mmlu": {
        "hf_direct": "UGRIP-LM-Polygraph/mmlu-direct",
        "eval_split": "test",
        "few_shot_split": "dev",  # mmlu-direct has dev/test/validation (no train)
        "process_fn": "process_output_mcq",
        "paper_config": "polygraph_eval_mmlu.yaml",
        "c2_extract": True,
        "note": "clean no-CoT few-shot anchor",
    },
    "gsm8k": {
        "hf_direct": "UGRIP-LM-Polygraph/gsm8k-direct",
        "eval_split": "test",
        "few_shot_split": "train",
        "process_fn": "process_output_number",
        "paper_config": "polygraph_eval_gsm8k.yaml",
        "c2_extract": False,  # paper gsm8k is CoT + sets normalize/ignore_regex
        "note": "paper C2 is CoT — NOT a no-CoT comparison",
    },
    # optional: no paper config, so C2 is skipped automatically.
    "medmcqa": {
        "hf_direct": "UGRIP-LM-Polygraph/medmcqa-direct",
        "eval_split": "validation",
        "few_shot_split": "train",
        "process_fn": "process_output_mcq",
        "paper_config": None,
        "c2_extract": False,
        "note": "no paper config (C0/C1 only)",
    },
}

# datasets actually included in the default grid (medmcqa optional — add if wanted)
GRID_DATASETS = ["mmlu", "gsm8k"]

ESTIMATORS = ["MaximumSequenceProbability", "Perplexity", "MeanTokenEntropy", "TokenSAR"]
ESTIMATOR_SHORT = {
    "MaximumSequenceProbability": "MSP",
    "Perplexity": "PPL",
    "MeanTokenEntropy": "MTE",
    "TokenSAR": "TokenSAR",
}

CONFIG_DIR = "./examples/configs"
EST_CONFIG = "diagnose_fewshot_estimators.yaml"
GEN_METRICS_CONFIG = "ugrip_benchmark_generation_metrics_acc.yaml"
PROCESS_SCRIPT = "instruct/output_processing_scripts/ugrip.py"


def _run_dir(condition: str, model: str, dataset: str) -> Path:
    return OUT_DIR / f"{condition}__{model}__{dataset}"


def _csv_path(dataset: str) -> Path:
    return OUT_DIR / "csv" / f"c1_fewshot_{dataset}.csv"


# --------------------------------------------------------------------------- #
# build: construct the C1 few-shot CSVs
# --------------------------------------------------------------------------- #

def build():
    """Build one few-shot CSV per dataset: K exemplars (from the train split)
    prepended to every eval-split question, written in dataset order so that the
    seeded subsample picks the same rows as the C0 run."""
    from datasets import get_dataset_split_names, load_dataset

    (OUT_DIR / "csv").mkdir(parents=True, exist_ok=True)
    for ds in GRID_DATASETS:
        info = DATASETS[ds]
        repo = info["hf_direct"]
        print(f"[build] {ds}: loading {repo} ...")

        # pick a few-shot source split that exists and is not the eval split
        splits = get_dataset_split_names(repo)
        fs_split = info["few_shot_split"]
        if fs_split not in splits:
            fs_split = next(
                (s for s in ("dev", "train", "validation") if s in splits and s != info["eval_split"]),
                None,
            ) or next((s for s in splits if s != info["eval_split"]), info["eval_split"])
            print(
                f"[build] {ds}: few_shot_split '{info['few_shot_split']}' not found; "
                f"using '{fs_split}' (available: {splits})"
            )
        few = load_dataset(repo, split=fs_split)
        # deterministic exemplar choice without Date/random-module surprises
        idx = list(range(min(N_SHOT, len(few))))
        prefix = ""
        for i in idx:
            q = str(few[i]["question"]).strip()
            a = str(few[i]["answer"]).strip()
            prefix += f"{q}\n{a}\n\n"

        test = load_dataset(repo, split=info["eval_split"])
        out = _csv_path(ds)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["question", "answer"])
            w.writeheader()
            for row in test:
                w.writerow(
                    {
                        "question": prefix + str(row["question"]).strip(),
                        "answer": row["answer"],
                    }
                )
        print(f"[build] wrote {out} ({len(test)} rows, {len(idx)}-shot prefix)")
    print(
        "\n[build] NOTE: exemplars are concatenated as '<question>\\n<answer>' pairs "
        "(dataset-agnostic). If the UGRIP `question` text already carries an "
        "instruction, the K-shot block sits in front of the first one — inspect a "
        "CSV row to confirm the format reads naturally."
    )


# --------------------------------------------------------------------------- #
# commands: emit polygraph_eval invocations for C0 / C1 / C2
# --------------------------------------------------------------------------- #

# `++` = add-or-override, so these are safe whether or not the target config
# already defines the key (ugrip defines use_claim_ue but not output_attentions;
# the paper configs define neither). seed is left at each config's default ([1]).
_COMMON = (
    f"--config-dir={CONFIG_DIR} "
    f"estimators={EST_CONFIG} "
    f"++use_claim_ue=false ++output_attentions=false batch_size=1"
)


def _process_overrides(fn: str) -> str:
    return (
        f"process_output_fn.path={PROCESS_SCRIPT} process_output_fn.fn_name={fn} "
        f"process_target_fn.path={PROCESS_SCRIPT} process_target_fn.fn_name=process_target"
    )


def _cmd(config_name: str, overrides: str, run_dir: Path) -> str:
    return (
        "uv run --python 3.11 scripts/polygraph_eval "
        f"--config-name={config_name} {_COMMON} "
        f"generation_metrics={GEN_METRICS_CONFIG} "
        f"{overrides} "
        f"hydra.run.dir={run_dir}"
    )


def commands():
    print("#!/bin/bash")
    print("set -e")
    print(f"# Phase-0 diagnostic runs. Outputs under {OUT_DIR}/")
    print(f"# subsample={SUBSAMPLE}, {N_SHOT}-shot, seed={SEED}\n")
    for model_short, model_cfg in MODELS.items():
        for ds in GRID_DATASETS:
            info = DATASETS[ds]
            fn = info["process_fn"]

            # C0 — direct 0-shot baseline
            c0 = _cmd(
                "polygraph_eval_ugrip.yaml",
                f"model={model_cfg} dataset={info['hf_direct']} "
                f"eval_split={info['eval_split']} {_process_overrides(fn)} "
                f"subsample_eval_dataset={SUBSAMPLE}",
                _run_dir("c0", model_short, ds),
            )
            # C1 — direct few-shot (CSV built by `build`)
            c1 = _cmd(
                "polygraph_eval_ugrip.yaml",
                f"model={model_cfg} dataset={_csv_path(ds)} "
                f"{_process_overrides(fn)} subsample_eval_dataset={SUBSAMPLE}",
                _run_dir("c1", model_short, ds),
            )
            print(f"# ---- {model_short} / {ds} ({info['note']}) ----")
            print(c0, "\n")
            print(c1, "\n")

            # C2 — paper config, run paper-native (only model + subsample +
            # estimators overridden). Its extraction is the paper config's own
            # (none for mmlu -> bare compare; regex for gsm8k), so C2's accuracy
            # axis is NOT directly comparable to C0/C1 — the rigorous comparison
            # is C0 vs C1. C2 shows how the paper pipeline behaves on our models.
            if info["paper_config"]:
                c2 = (
                    "uv run --python 3.11 scripts/polygraph_eval "
                    f"--config-name={info['paper_config']} {_COMMON} "
                    f"model={model_cfg} subsample_eval_dataset={SUBSAMPLE} "
                    f"hydra.run.dir={_run_dir('c2', model_short, ds)}"
                )
                print(c2, "\n")
    print(
        "# NOTE: C0 vs C1 is the rigorous comparison (identical extraction). C2 "
        "runs the paper config paper-native; gsm8k-C2 is CoT, not a no-CoT point."
    )


# --------------------------------------------------------------------------- #
# analyze: load .man files and print the comparison table
# --------------------------------------------------------------------------- #

def _load_man(path: Path):
    import torch

    return torch.load(str(path), weights_only=False)


def _acc(man) -> float:
    import numpy as np

    gm = man.get("gen_metrics") or {}
    arr = gm.get(("sequence", "Accuracy"))
    if arr is None:
        return float("nan")
    arr = np.asarray(arr, dtype=float)
    return float(arr[~np.isnan(arr)].mean()) if arr.size else float("nan")


def _prr(man, estimator: str, max_rej: str) -> float:
    """max_rej: 'prr' (1.0) or 'prr_0.5'. Returns normalized PRR or nan."""
    metrics = man.get("metrics") or {}
    key = ("sequence", estimator, "Accuracy", f"{max_rej}_normalized")
    val = metrics.get(key)
    return float(val) if val is not None else float("nan")


def analyze():
    mans = sorted(OUT_DIR.glob("*/*.man"))
    if not mans:
        print(f"No .man files under {OUT_DIR}/*/  — run the `commands` stage first.")
        return

    # results[(model, dataset)][condition] = {"acc":x, "MSP":..., ...}
    results = {}
    for p in mans:
        parent = p.parent.name  # <condition>__<model>__<dataset>
        parts = parent.split("__")
        if len(parts) != 3:
            print(f"  skip {p} (unexpected dir name {parent!r})")
            continue
        cond, model, ds = parts
        man = _load_man(p)
        row = {"acc": _acc(man)}
        for est in ESTIMATORS:
            row[ESTIMATOR_SHORT[est]] = _prr(man, est, "prr_0.5")
        results.setdefault((model, ds), {})[cond] = row

    cols = ["acc"] + [ESTIMATOR_SHORT[e] for e in ESTIMATORS]
    order = ["c0", "c1", "c2"]

    def fmt(v):
        return "  n/a " if v != v else f"{v:6.3f}"  # v!=v => nan

    print("\nPRR@0.5 (normalized) + accuracy — Δ vs C0 in parentheses\n")
    for (model, ds) in sorted(results.keys()):
        conds = results[(model, ds)]
        base = conds.get("c0", {})
        print(f"== {model} / {ds} ==")
        header = "cond   " + "".join(f"{c:>10}" for c in cols)
        print(header)
        for cond in order:
            if cond not in conds:
                continue
            r = conds[cond]
            cells = []
            for c in cols:
                v = r.get(c, float("nan"))
                s = fmt(v)
                b = base.get(c)
                if cond != "c0" and b is not None and b == b and v == v:
                    s += f"({v - b:+.3f})"
                else:
                    s += " " * 8
                cells.append(f"{s:>18}")
            print(f"{cond:<6}" + "".join(cells))
        print()
    print(
        "Read: if C1/C2 lift MMLU accuracy + PRR well above C0, direct-mode "
        "weakness is the 0-shot/verbose-generation setup. If they barely move, "
        "the cause is deeper (instruct calibration / chat template / harness).",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="build the C1 few-shot CSVs")
    sub.add_parser("commands", help="print polygraph_eval commands for C0/C1/C2")
    sub.add_parser("analyze", help="tabulate results from the .man files")
    args = ap.parse_args()

    if args.cmd == "build":
        build()
    elif args.cmd == "commands":
        commands()
    elif args.cmd == "analyze":
        analyze()


if __name__ == "__main__":
    main()
