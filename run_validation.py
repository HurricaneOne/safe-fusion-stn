#!/usr/bin/env python3
"""
run_validation.py
-----------------
Run the three Perturb-seq validation tests on one or more imputers.

Usage:
  # verify the harness on synthetic data (no real deps needed):
  python run_validation.py --synthetic --imputers identity oversmooth conservative

  # real run: your .h5ad must contain guide_target in .obs and (for Test 3)
  # a protein matrix in .obsm['protein'] with .uns['protein_names'] /
  # .uns['rna_for_protein'].  scVI must be included as a baseline.
  python run_validation.py --h5ad data.h5ad --imputers safefusion scvi --tau 0.081

Output: one row per (imputer x test) with AUCs and fill-matched precision/recall.
Save with --out results.csv, then hand the CSV over for the write-up.
"""
import argparse, json, sys
import numpy as np
import pandas as pd

import imputers as IMP
from validation import run_all


def load(args):
    if args.synthetic:
        from synth_perturbseq import make_synth
        return make_synth(seed=args.seed)
    import anndata as ad
    from scipy import sparse
    a = ad.read_h5ad(args.h5ad)
    assert "guide_target" in a.obs, "need adata.obs['guide_target'] (run build_eccite_harness.py)"
    if sparse.issparse(a.X):
        a.X = np.asarray(a.X.todense(), dtype="float32")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.081)
    ap.add_argument("--imputers", nargs="+", default=["identity", "conservative"])
    ap.add_argument("--extra_imputer", nargs=2, action="append", metavar=("NAME", "NPY"),
                    default=[], help="score a saved cells x genes matrix, e.g. "
                    "--extra_imputer scvi output/scVI/eccite_filtered/imputed.npy")
    ap.add_argument("--binary_panel", action="store_true",
                    help="protein panel is binary 0/1 (use >0 as detected)")
    ap.add_argument("--min_ctrl_expr_frac", type=float, default=0.05,
                    help="Test 1: min fraction of control cells expressing a target "
                    "to use it (lower for surface genes like CD86)")
    ap.add_argument("--out")
    args = ap.parse_args()

    adata = load(args)
    theta_abs = 0.0 if args.binary_panel else None
    print(f"# data: {adata.n_obs} cells x {adata.n_vars} genes | "
          f"zero_frac={(np.asarray(adata.X)==0).mean():.3f} | tau={args.tau}\n")

    # assemble the list of (name, imputed_matrix_fn)
    jobs = []
    for name in args.imputers:
        if name not in IMP.REGISTRY:
            print(f"!! unknown imputer '{name}', skipping"); continue
        jobs.append((name, lambda X, n=name: IMP.REGISTRY[n](X)))
    for name, path in args.extra_imputer:
        def _load_npy(X, p=path):
            import glob as _g, os as _o
            if _o.path.isdir(p):
                cand = sorted(_g.glob(_o.path.join(p, "**", "*.npy"), recursive=True))
            else:
                cand = [p]
            tried = []
            for f in cand:
                try:
                    m = np.load(f)
                except Exception:
                    continue
                if m.shape == (X.shape[1], X.shape[0]):
                    m = m.T
                tried.append(f"{_o.path.basename(f)}{m.shape}")
                if m.shape == X.shape:
                    print(f"   using teacher matrix: {f}")
                    return m.astype("float32")
            raise AssertionError(f"no cells x genes .npy matching {X.shape} under {p}. "
                                 f"found: {tried}")
        jobs.append((name, _load_npy))

    rows = []
    for name, fn in jobs:
        print(f"## running imputer: {name}")
        try:
            imputed = fn(np.asarray(adata.X, dtype="float32"))
        except NotImplementedError as e:
            print(f"   SKIPPED ({e})\n"); continue
        except Exception as e:
            print(f"   ERROR ({type(e).__name__}: {e})\n"); continue
        for res in run_all(adata, imputed, tau=args.tau, theta_abs=theta_abs,
                           min_ctrl_expr_frac=args.min_ctrl_expr_frac):
            res = {"imputer": name, **res}
            rows.append(res)
            print("   " + json.dumps(res))
        print()

    if rows:
        df = pd.DataFrame(rows)
        cols = ["imputer", "test", "roc_auc", "pr_auc", "prec@tau", "rec@tau",
                "f1@tau", "ko_fill_rate", "lfc_corr", "signal_retained",
                "target_roc_auc", "target_pr_auc", "n_pos", "n_neg", "n_tfs"]
        df = df[[c for c in cols if c in df.columns]]
        print("\n===== SUMMARY =====")
        print(df.to_string(index=False))
        if args.out:
            df.to_csv(args.out, index=False)
            print(f"\nwrote {args.out}")
    else:
        print("no results (did every imputer skip/error?)")


if __name__ == "__main__":
    sys.exit(main())
