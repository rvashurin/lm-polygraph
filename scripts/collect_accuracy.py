"""Collect overall accuracy (and other generation-metric means) across a set of
saved polygraph_eval runs.

Each run's `.man` file stores the per-sample generation-metric arrays and, for
runs produced after the accuracy-reporting change, a `gen_metric_means` dict with
the overall mean of each metric. This script reads that mean directly (falling
back to computing it from the per-sample array for older `.man` files) and prints
a table over every `.man` found — the "total accuracy for all runs" surface.

Unlike scripts/test_answer_extraction.py, this reflects the *actual* extraction
used in the run (it reads the saved AccuracyMetric result, it does not re-extract).

Usage:
    python scripts/collect_accuracy.py <dir> [<dir> ...]     # recurse for *.man
    python scripts/collect_accuracy.py run1.man run2.man     # explicit files
"""

import sys
from pathlib import Path

import numpy as np
import torch


def _iter_man_paths(args):
    for a in args:
        p = Path(a)
        if p.is_dir():
            yield from sorted(p.rglob("*.man"))
        elif p.is_file():
            yield p
        else:
            print(f"  skip {a}: not a file or directory")


def _means(man) -> dict:
    """Return {(level, name): mean}. Prefer the persisted gen_metric_means; fall
    back to computing from the per-sample gen_metrics arrays (older .man)."""
    persisted = man.get("gen_metric_means")
    if persisted:
        return dict(persisted)
    out = {}
    for key, vals in (man.get("gen_metrics") or {}).items():
        arr = np.asarray(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        out[key] = float(finite.mean()) if finite.size else float("nan")
    return out


def _n_samples(man) -> int:
    gm = man.get("gen_metrics") or {}
    for vals in gm.values():
        return len(vals)
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths = list(_iter_man_paths(sys.argv[1:]))
    if not paths:
        print("No .man files found.")
        sys.exit(1)

    rows = []
    metric_names = []  # preserve first-seen order across runs
    for p in paths:
        try:
            man = torch.load(str(p), weights_only=False)
        except Exception as e:
            print(f"  skip {p}: load failed ({e})")
            continue
        means = _means(man)
        # key by metric name (ignore level; sequence-level is what we report)
        named = {}
        for (level, name), val in means.items():
            named[name] = val
            if name not in metric_names:
                metric_names.append(name)
        rows.append((str(p), _n_samples(man), named))

    if not rows:
        print("No loadable .man files.")
        sys.exit(1)

    # put Accuracy first if present
    if "Accuracy" in metric_names:
        metric_names.remove("Accuracy")
        metric_names.insert(0, "Accuracy")

    label_w = max(len(r[0]) for r in rows)
    header = f"{'run':<{label_w}}  {'n':>6}  " + "  ".join(f"{m:>10}" for m in metric_names)
    print(header)
    print("-" * len(header))
    for path, n, named in rows:
        cells = []
        for m in metric_names:
            v = named.get(m)
            cells.append("     n/a  " if v is None else f"{v:10.4f}")
        print(f"{path:<{label_w}}  {n:>6}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
