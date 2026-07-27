#!/usr/bin/env python3
"""
inspect_eccite.py  --  read-only. Reports what's inside the ECCITE files so we
know how to feed them to the Perturb-seq harness. Changes nothing.
Run from the repo root:  python inspect_eccite.py
"""
import os, sys
try:
    import anndata as ad, numpy as np, pandas as pd
except Exception as e:
    print("!! could not import anndata/numpy/pandas in this environment.")
    print("   error:", e)
    print("   -> run this inside the conda env that has the repo's deps.")
    sys.exit(1)

CANDIDATES = ["data/eccite_filtered.h5ad", "data/GSE153056_eccite.h5ad",
              "data/GSE153056_eccite_raw.h5ad"]
GUIDE_HINTS = ["guide", "grna", "grna", "perturb", "target", "gene", "sgrna",
               "knockout", "ko", "crispr", "condition", "gene_target"]
PROT_HINTS  = ["protein", "adt", "citeseq", "cite", "ab", "antibody"]

def show(path):
    print("\n" + "=" * 70)
    print("FILE:", path)
    if not os.path.exists(path):
        print("   (not present)"); return
    print("   size: %.1f MB" % (os.path.getsize(path) / 1e6))
    a = ad.read_h5ad(path)
    print(f"   shape: {a.n_obs} cells x {a.n_vars} genes")
    print(f"   X dtype: {a.X.dtype}, looks like counts: "
          f"{np.allclose(np.asarray(a.X[:50].todense() if hasattr(a.X,'todense') else a.X[:50]) % 1, 0)}")

    print("\n   .obs columns:")
    for c in a.obs.columns:
        col = a.obs[c]
        nun = col.nunique()
        flag = "  <-- possible GUIDE/PERTURBATION" if any(h in c.lower() for h in GUIDE_HINTS) else ""
        ex = list(pd.Series(col.unique()).astype(str)[:6])
        print(f"      - {c}  (dtype={col.dtype}, {nun} unique){flag}")
        if nun <= 40 or flag:
            print(f"          e.g. {ex}")

    print("\n   .var index name:", a.var.index.name, "| example genes:",
          list(a.var_names[:8]))
    print("   .obsm keys:", {k: v.shape for k, v in a.obsm.items()})
    print("   .layers keys:", list(a.layers.keys()))
    print("   .uns keys:", list(a.uns.keys()))
    # protein hints anywhere
    prot_here = [k for k in list(a.obsm.keys()) + list(a.uns.keys())
                 if any(h in k.lower() for h in PROT_HINTS)]
    if prot_here:
        print("   PROTEIN-like keys:", prot_here)

for p in CANDIDATES:
    try:
        show(p)
    except Exception as e:
        print(f"   ERROR reading {p}: {type(e).__name__}: {e}")

# peek at the validation panel (protein alignment for Test 3)
panel = "data/eccite_validation_panel.csv"
print("\n" + "=" * 70)
print("VALIDATION PANEL:", panel)
if os.path.exists(panel):
    df = pd.read_csv(panel, nrows=5)
    print("   columns:", list(df.columns))
    print(df.head().to_string())
else:
    print("   (not present)")
