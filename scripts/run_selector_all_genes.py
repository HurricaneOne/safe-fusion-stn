#!/usr/bin/env python3
"""Run calibrated_selective_fill.py scoring EVERY gene in --truth, without
passing ~1200+ gene names through the OS command line (Windows cmd.exe caps
around 8191 chars, which a full --score-gene list blows past).

This sets sys.argv in-process and calls calibrated_selective_fill.main()
directly, so the gene list never goes through a shell at all.

Usage (same flags as calibrated_selective_fill.py, EXCEPT no --score-gene --
it is filled in automatically from every gene in --truth):

  python scripts/run_selector_all_genes.py \
    --corrupted <deploy>/hybrid.h5ad --truth <fold>/truth.h5ad \
    --coordinates <deploy>/coordinates.parquet --splits <deploy>/splits.parquet \
    --fusion-contract <deploy>/methods/safe_fusion \
    --teacher-contract <deploy>/methods/gene_median \
    --teacher-contract <deploy>/methods/svd_impute \
    --teacher-contract <deploy>/methods/graph_smooth \
    --teacher-contract <deploy>/methods/magic_inductive \
    --teacher-contract <deploy>/methods/scvi_inductive \
    --output-dir <deploy>/selector --fit-split development \
    --architecture mlp --budget-mode apply_topk \
    --budgets 0.01 0.05 0.1 --curve-points 2 --seed 1729
"""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrated_selective_fill as csf


def main() -> None:
    args = sys.argv[1:]
    if "--score-gene" in args:
        raise SystemExit("Do not pass --score-gene here; it is filled in from every gene in --truth.")
    if "--truth" not in args:
        raise SystemExit("--truth is required (same as calibrated_selective_fill.py).")
    truth_path = args[args.index("--truth") + 1]
    genes = ad.read_h5ad(truth_path, backed="r").var_names.astype(str).tolist()
    print(f"[run_selector_all_genes] scoring {len(genes)} genes from {truth_path}")
    sys.argv = [sys.argv[0], *args, "--score-gene", *genes]
    csf.main()


if __name__ == "__main__":
    main()
