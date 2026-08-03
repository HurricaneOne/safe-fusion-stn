"""
test3_threshold_sweep.py
------------------------
Checks whether the Test 3 cross-modal result is stable across the choice of
"protein detectable" threshold. Rebuilds CLR-normalized PDL1 labels at several
cutoffs and reports ROC-AUC / PR-AUC for each imputer at each threshold.

Self-contained: computes CLR, applies threshold, builds (score, label) pairs
exactly as test3_crossmodal does (among cells where CD274 RNA == 0), and scores
each imputer matrix. No harness rebuild needed.

Run:
  python test3_threshold_sweep.py \
      --panel data/eccite_validation_panel_real.csv \
      --rna data/GSE153056_eccite.h5ad \
      --imputer safefusion safefusion_scores_full.npy \
      --imputer scvi output/scVI/GSE153056_eccite/ECCITE/ECCITE.npy \
      --imputer identity IDENTITY
"""
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from sklearn.metrics import roc_auc_score, average_precision_score

PROT = "PDL1"
GENE = "CD274"


def clr_for_protein(panel_df, protein):
    """Return a Series indexed by cell_id of CLR-normalized values for one
    protein, with CLR computed per cell across all proteins."""
    wide = panel_df.pivot_table(index="cell_id", columns="protein",
                                values="measurement", aggfunc="first")
    log1p = np.log1p(wide.values.astype("float64"))
    gm_log = log1p.mean(axis=1, keepdims=True)
    clr = log1p - gm_log
    clr_df = pd.DataFrame(clr, index=wide.index, columns=list(wide.columns))
    return clr_df[protein]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/eccite_validation_panel_real.csv")
    ap.add_argument("--rna", default="data/GSE153056_eccite.h5ad")
    ap.add_argument("--imputer", nargs=2, action="append", metavar=("NAME", "PATH"),
                    help="name and .npy path; use PATH=IDENTITY for the identity baseline")
    ap.add_argument("--quantiles", default="0.3,0.4,0.5,0.6,0.7")
    a = ap.parse_args()

    rna = ad.read_h5ad(a.rna)
    X = rna.X
    Xg = (np.asarray(X[:, list(rna.var_names).index(GENE)].todense()).ravel()
          if sparse.issparse(X)
          else np.asarray(X[:, list(rna.var_names).index(GENE)]).ravel())
    rna_ids = list(rna.obs_names)

    panel = pd.read_csv(a.panel)
    clr = clr_for_protein(panel, PROT)   # indexed by cell_id

    # align CLR protein values to RNA cell order
    clr_aligned = clr.reindex(rna_ids).values   # NaN where no protein match

    # cells where CD274 RNA is zero AND we have a protein value
    zero = (Xg == 0) & ~np.isnan(clr_aligned)
    zidx = np.where(zero)[0]
    prot_at_zero = clr_aligned[zidx]
    print(f"CD274 RNA-zero cells with protein data: {len(zidx)}")
    print(f"CLR range at those cells: [{prot_at_zero.min():.3f}, {prot_at_zero.max():.3f}]\n")

    # load imputers
    imputers = {}
    gene_j = list(rna.var_names).index(GENE)
    for name, path in (a.imputer or []):
        if path == "IDENTITY":
            imputers[name] = Xg  # identity score at CD274 = raw (all zero here)
        else:
            M = np.load(path)
            imputers[name] = M[:, gene_j]

    quants = [float(q) for q in a.quantiles.split(",")]
    print(f"{'threshold':>10s} {'%detect':>8s} " +
          " ".join(f"{n+'_ROC':>16s}" for n in imputers))
    for q in quants:
        thr = np.quantile(prot_at_zero, q)
        labels = (prot_at_zero > thr).astype(int)
        pct = 100 * labels.mean()
        cells_roc = []
        for name, col in imputers.items():
            scores = col[zidx]
            try:
                roc = roc_auc_score(labels, scores)
            except ValueError:
                roc = float("nan")
            cells_roc.append(f"{roc:16.4f}")
        print(f"{q:10.2f} {pct:7.1f}% " + " ".join(cells_roc))

    print("\nAlso: 'above baseline' cutoff (CLR > 0):")
    labels0 = (prot_at_zero > 0).astype(int)
    pct0 = 100*labels0.mean()
    row = []
    for name, col in imputers.items():
        try:
            row.append(f"{roc_auc_score(labels0, col[zidx]):16.4f}")
        except ValueError:
            row.append(f"{'nan':>16s}")
    print(f"{'CLR>0':>10s} {pct0:7.1f}% " + " ".join(row))


if __name__ == "__main__":
    main()
