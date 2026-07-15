"""Run answer extractors over saved .man files and dump a CSV comparing
the ORIGINAL extractor (pre-changes), the NEW general extractor, and the
NEW per-task extractor.

Usage:
    python scripts/test_answer_extraction.py man/ man_extraction.csv

The CSV has columns:
  file, task, processor, raw_output (truncated), target, target_processed,
  original_extraction, new_general, new_per_task,
  original_match, new_general_match, new_per_task_match
"""

import csv
import hashlib
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "configs" / "instruct" / "output_processing_scripts"))
import ugrip


# --- ORIGINAL extractor (pre-changes), inlined for an apples-to-apples diff ---
_ORIG_GEMMA = re.compile(r"<end_of_turn>")
_ORIG_REASONING = re.compile(r"(?s).*### Answer:\s*")
_ORIG_PAREN = re.compile(r"\)")
_ORIG_INT = re.compile(r"\d+")


def original_process_output(output: str) -> str:
    o = _ORIG_GEMMA.sub("", output)
    o = _ORIG_REASONING.sub("", o)
    o = _ORIG_PAREN.sub("", o)
    m = _ORIG_INT.search(o)
    if m:
        return str(m.group())
    return str(o)


def original_process_target(t: str) -> str:
    return str(t)


def detect_task(name: str) -> str:
    n = name.lower()
    if "mmlu" in n:
        return "mmlu"
    if "medmcqa" in n:
        return "medmcqa"
    if "gsm8k" in n:
        return "gsm8k"
    return "unknown"


def pick_new_per_task(task: str):
    if task in ("mmlu", "medmcqa"):
        return ugrip.process_output_mcq, "process_output_mcq"
    if task == "gsm8k":
        return ugrip.process_output_number, "process_output_number"
    return ugrip.process_output, "process_output"


def load_man(path: Path):
    return torch.load(str(path), weights_only=False)


def extract_pairs(res):
    stats = res.get("stats") or {}
    greedy = stats.get("greedy_texts") or stats.get("greedy_texts_full")
    target = res.get("target_texts")
    if target is None:
        target = stats.get("target_texts")
    if greedy is None or target is None:
        return []
    return list(zip(greedy, target))


def fingerprint(pairs):
    """Hash of the (greedy, target) pairs to identify duplicate files."""
    h = hashlib.sha256()
    for g, t in pairs:
        h.update(repr(g).encode("utf-8", errors="replace"))
        h.update(b"\x1e")
        h.update(repr(t).encode("utf-8", errors="replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    in_dir = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])

    files = sorted(in_dir.glob("*.man"))
    if not files:
        print(f"No .man files in {in_dir}")
        sys.exit(1)

    rows = []
    summary = []
    seen_fingerprints: dict[str, str] = {}
    for f in files:
        task = detect_task(f.name)
        proc_pt, proc_pt_name = pick_new_per_task(task)
        try:
            res = load_man(f)
        except Exception as e:
            print(f"  skip {f.name}: load failed ({e})")
            continue
        pairs = extract_pairs(res)
        if not pairs:
            print(f"  skip {f.name}: no greedy_texts/target_texts")
            continue

        fp = fingerprint(pairs)
        dup_of = seen_fingerprints.get(fp)
        if dup_of:
            print(f"  DUP  {f.name}  (same data as {dup_of})")
        else:
            seen_fingerprints[fp] = f.name

        orig_correct = 0
        new_general_correct = 0
        new_pt_correct = 0
        for greedy, target in pairs:
            g = str(greedy)
            orig_ans = original_process_output(g)
            orig_tgt = original_process_target(str(target))

            new_general_ans = ugrip.process_output(g)
            new_pt_ans = proc_pt(g)
            new_tgt = ugrip.process_target(str(target))

            orig_match = int(orig_ans.strip() == orig_tgt.strip())
            new_general_match = int(new_general_ans.strip() == new_tgt.strip())
            new_pt_match = int(new_pt_ans.strip() == new_tgt.strip())

            orig_correct += orig_match
            new_general_correct += new_general_match
            new_pt_correct += new_pt_match

            rows.append({
                "file": f.name,
                "task": task,
                "duplicate_of": dup_of or "",
                "processor": proc_pt_name,
                "raw_output": g[:500],
                "target": str(target),
                "target_processed_new": new_tgt,
                "original_extraction": orig_ans[:200],
                "new_general": new_general_ans[:200],
                "new_per_task": new_pt_ans[:200],
                "original_match": orig_match,
                "new_general_match": new_general_match,
                "new_per_task_match": new_pt_match,
            })

        n = len(pairs)
        summary.append({
            "file": f.name,
            "task": task,
            "duplicate_of": dup_of or "",
            "processor": proc_pt_name,
            "n": n,
            "original_acc": orig_correct / n if n else 0.0,
            "new_general_acc": new_general_correct / n if n else 0.0,
            "new_per_task_acc": new_pt_correct / n if n else 0.0,
        })
        flag = "  DUP" if dup_of else "     "
        print(
            f"{flag}  {f.name}: n={n} "
            f"orig={orig_correct/n:.3f} "
            f"new_gen={new_general_correct/n:.3f} "
            f"new_pt={new_pt_correct/n:.3f}"
        )

    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote per-row CSV: {out_csv} ({len(rows)} rows)")

    summary_csv = out_csv.with_name(out_csv.stem + "_summary.csv")
    with summary_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"Wrote summary CSV:   {summary_csv}")

    dups = [s for s in summary if s["duplicate_of"]]
    if dups:
        print(f"\nDuplicate files ({len(dups)}):")
        for s in dups:
            print(f"  {s['file']}  == {s['duplicate_of']}")


if __name__ == "__main__":
    main()
