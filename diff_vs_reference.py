#!/usr/bin/env python3
"""Diff a refit run's CD274 method_summary against the shipped manuscript reference.

Usage:
    python diff_vs_reference.py <run_dir> [--reference <dir>] [--out <csv>]

<run_dir> is the OUTDIR produced by refit_figure2.sh (it must contain
evaluation_cd274/method_summary.csv). The reference defaults to the shipped
benchmark evaluation that came in the LFS tarball.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

REF_DEFAULT = Path(
    "artifacts/paper_evidence/papalexi_crossmodal/benchmark/evaluation_cd274/method_summary.csv"
)
METRICS = [
    ("spearman", "Spearman"),
    ("mean_roc_auc_q30_q70", "Mean AUROC q30-q70"),
    ("mean_pr_auc_q30_q70", "Mean PR-AUC q30-q70"),
    ("mean_protein_enrichment_fill01_fill20", "PD-L1 enrichment fill 1-20%"),
]
LABELS = {
    "mlp_safe_fusion": "Safe Fusion (MLP-gated)",
    "fused_component": "Fused value",
    "svd": "SVD",
    "weighted_knn": "Weighted kNN",
    "scvi": "scVI",
    "library_size": "Library size",
}
ORDER = list(LABELS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--reference", type=Path, default=REF_DEFAULT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = pd.read_csv(args.run_dir / "evaluation_cd274" / "method_summary.csv").set_index("method")
    ref = pd.read_csv(args.reference).set_index("method")

    rows = []
    for metric, mname in METRICS:
        if metric not in run.columns or metric not in ref.columns:
            continue
        for m in ORDER:
            if m not in run.index or m not in ref.index:
                continue
            mine = float(run.loc[m, metric])
            base = float(ref.loc[m, metric])
            rows.append(
                {
                    "metric": mname,
                    "method": LABELS[m],
                    "manuscript": base,
                    "refit": mine,
                    "diff": mine - base,
                    "abs_diff": abs(mine - base),
                }
            )
    df = pd.DataFrame(rows)
    out = args.out or (args.run_dir / "diff_cd274.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nWritten: {out}")
    print(f"max |diff| = {df['abs_diff'].max():.6f}")


if __name__ == "__main__":
    main()
