#!/usr/bin/env python3
"""
inspect_outputs.py -- read-only. Figures out whether a reusable safe-fusion
score matrix is already on disk (and whether posterior variance exists to
rebuild it), and where the repo saves its scores. Changes nothing.
Run from repo root:  python inspect_outputs.py
"""
import os, sys, glob, json
try:
    import numpy as np
except Exception as e:
    print("!! numpy import failed -- activate safefusion_env first. err:", e); sys.exit(1)

def eccite_dims():
    try:
        import anndata as ad
        for p in ["data/eccite_filtered.h5ad", "data/GSE153056_eccite.h5ad"]:
            if os.path.exists(p):
                a = ad.read_h5ad(p)
                return p, a.n_obs, a.n_vars
    except Exception as e:
        print("   (could not read eccite dims:", e, ")")
    return None, None, None

path, ncell, ngene = eccite_dims()
print("ECCITE reference:", path, "| cells:", ncell, "| genes:", ngene)
tgt = {ncell * ngene if ncell else -1: "cells*genes (FULL MATRIX)",
       ncell: "n_cells", ngene: "n_genes"}

def describe_npy(fp):
    try:
        arr = np.load(fp, allow_pickle=True)
        shp = getattr(arr, "shape", None)
        note = ""
        if shp and ncell:
            if shp == (ncell, ngene) or shp == (ngene, ncell):
                note = "  <== CELLS x GENES full matrix"
            elif arr.size == ncell * ngene:
                note = "  <== size matches cells*genes"
        print(f"      {os.path.basename(fp)}: shape={shp} dtype={getattr(arr,'dtype',None)}{note}")
    except Exception as e:
        print(f"      {os.path.basename(fp)}: (load failed: {e})")

print("\n===== output/ method folders =====")
for d in sorted(glob.glob("output/*")):
    if not os.path.isdir(d): continue
    files = os.listdir(d)
    print(f"\n  [{d}]  ({len(files)} items)")
    for f in files[:12]:
        fp = os.path.join(d, f)
        if f.endswith((".npy", ".npz")): describe_npy(fp)
        elif f.endswith(".h5ad"):
            try:
                import anndata as ad
                a = ad.read_h5ad(fp)
                print(f"      {f}: h5ad {a.n_obs}x{a.n_vars} | layers={list(a.layers.keys())} | obsm={list(a.obsm.keys())}")
            except Exception as e:
                print(f"      {f}: h5ad (read failed: {e})")
        else:
            print(f"      {f}")

print("\n===== output_eccite_results/ =====")
for f in sorted(glob.glob("output_eccite_results/*.npy")):
    describe_npy(f)
psum = "output_eccite_results/precision_summary.json"
if os.path.exists(psum):
    with open(psum) as fh: j = json.load(fh)
    print("  precision_summary.json top-level keys:", list(j.keys())[:30])
    flat = json.dumps(j).lower()
    print("  mentions 'safe':", "safe" in flat, "| mentions 'fusion':", "fusion" in flat)
for c in ["method_operating_points.csv", "method_weight_summary.csv", "weights.json"]:
    fp = os.path.join("output_eccite_results", c)
    if os.path.exists(fp):
        print(f"\n  --- {c} ---")
        print("   " + open(fp).read()[:800].replace("\n", "\n   "))

print("\n===== code scan: where are scores computed/saved? =====")
KEYS = ["safe_fusion", "safe-fusion", "np.save", ".to_csv", "write_h5ad",
        "conf_quantile", "variance", "posterior", "def safe", "s_safe",
        "coverage", "fallback", "latent_score_mode"]
for src in ["precision_validation_experiments.py", "fusion/eval.py",
            "fusion/train.py", "fusion/models.py", "scripts/run_fusion_methods.py"]:
    if not os.path.exists(src): continue
    print(f"\n  [{src}]")
    for i, line in enumerate(open(src, encoding="utf-8", errors="ignore"), 1):
        low = line.lower()
        if any(k.lower() in low for k in KEYS):
            print(f"    {i:4d}: {line.rstrip()[:110]}")
