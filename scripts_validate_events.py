"""Evaluate event predictions against manually verified ground-truth JSON.

Usage: python scripts_validate_events.py --predictions predictions.json --ground-truth labels.json --tolerance 2.0
Both JSON files must be lists of {"timestamp": seconds, "label": class} objects.
Matches are one-to-one per class within the configured timestamp tolerance.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def score(predictions, truth, tolerance):
    labels = sorted({str(x["label"]) for x in predictions + truth})
    report = {}
    for label in labels:
        preds = sorted([float(x["timestamp"]) for x in predictions if x["label"] == label])
        refs = sorted([float(x["timestamp"]) for x in truth if x["label"] == label])
        used = set()
        tp = 0
        for p in preds:
            options = [(abs(p-r), i) for i, r in enumerate(refs)
                       if i not in used and abs(p-r) <= tolerance]
            if options:
                _, i = min(options)
                used.add(i)
                tp += 1
        fp, fn = len(preds) - tp, len(refs) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        report[label] = {"tp": tp, "fp": fp, "fn": fn,
                         "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--tolerance", type=float, default=2.0)
    args = parser.parse_args()
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    print(json.dumps({"tolerance_seconds": args.tolerance,
                      "per_class": score(predictions, truth, args.tolerance)}, indent=2))

if __name__ == "__main__":
    main()
