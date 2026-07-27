#!/usr/bin/env python3
"""
build_eccite_harness.py
-----------------------
Convert the repo's ECCITE-seq data into the Perturb-seq harness format, and
report (a) whether each test has signal and (b) what the saved teacher matrices
look like so we know which one to score.

Produces <out>.h5ad with:
  X                          raw counts (dense)
  obs['guide_target']        gene symbol of the perturbation, 'NT' for controls
  obsm['protein']            cells x n_proteins, binary detection, NaN if unmeasured
  uns['protein_names']       ['prot_<GENE>', ...]
  uns['rna_for_protein']     {'prot_<GENE>': '<GENE>'}
  (uns['tf_targets'] is left empty; Test 2 needs curated downstream modules.)

Usage:
  python build_eccite_harness.py --h5ad data/eccite_filtered.h5ad \
      --panel data/eccite_validation_panel.csv --out eccite_harness.h5ad
"""
import argparse, glob, os, re
import numpy as np, pandas as pd, anndata as ad
from scipy import sparse

# some anndata versions refuse to write newer 'nullable string' arrays by default
try:
    ad.settings.allow_write_nullable_strings = True
except Exception:
    pass


def strip_guide(v):
    """'CD86g1'->'CD86', 'STAT2g2'->'STAT2', 'control'/'NT*'->'NT'."""
    s = str(v)
    if s.lower() in ("control", "ctrl", "nt", "none", "non-targeting"):
        return "NT"
    s2 = re.sub(r"g\d+$", "", s)          # drop trailing guide index
    if s2.upper().startswith("NT"):
        return "NT"
    return s2


def densify(X):
    return np.asarray(X.todense() if sparse.issparse(X) else X, dtype="float32")


# Literature-grounded downstream modules for the interferon / JAK-STAT
# regulators in this ECCITE screen. Genes are standard HGNC symbols; the builder
# keeps only those present in the data. These are canonical interferon-stimulated
# genes (ISGs) and MHC/antigen-presentation targets known to be suppressed when
# the upstream regulator is knocked down.
CURATED_MODULES = {
    "STAT1": ["IRF1", "GBP1", "GBP2", "GBP4", "GBP5", "TAP1", "PSMB8", "PSMB9",
              "B2M", "WARS", "IDO1", "CXCL10", "SOCS1", "UBE2L6",
              "HLA-A", "HLA-B", "HLA-C"],
    "STAT2": ["ISG15", "IFI6", "IFI44", "IFI44L", "MX1", "MX2", "OAS1", "OAS2",
              "OAS3", "IFIT1", "IFIT3", "RSAD2", "USP18", "IRF7", "STAT1"],
    "IRF1":  ["TAP1", "PSMB9", "GBP1", "GBP2", "CXCL10", "CIITA", "B2M",
              "STAT1", "PSMB8"],
    "IRF7":  ["ISG15", "MX1", "OAS1", "IFIT1", "RSAD2", "IFI6", "USP18"],
    "JAK2":  ["STAT1", "SOCS1", "SOCS3", "CISH", "IRF1", "GBP1"],
    "IFNGR1": ["STAT1", "IRF1", "GBP1", "GBP2", "TAP1", "PSMB8", "PSMB9",
               "CIITA", "B2M", "CXCL10"],
    "IFNGR2": ["STAT1", "IRF1", "GBP1", "GBP2", "TAP1", "PSMB8", "PSMB9",
               "CIITA", "B2M", "CXCL10"],
    "STAT3": ["SOCS3", "BCL3", "MYC", "CISH", "JUNB"],
    "SPI1":  ["CD86", "ITGAM", "CSF1R", "TLR4"],
}


def build_tf_targets(targets, var_names, guide_series, name_to_col, X,
                     min_ko_cells=20, min_module_genes=3):
    """Populate downstream modules for regulators present in the data, keeping
    only module genes that exist in the matrix and regulators with enough KO
    cells. Returns {regulator: [target genes]} for Test 2."""
    out = {}
    print("\nTest 2 module assignment (curated interferon / JAK-STAT targets):")
    for tf in sorted(targets):
        if tf not in CURATED_MODULES:
            continue
        ko_cells = int((guide_series == tf).sum())
        present = [g for g in CURATED_MODULES[tf] if g in var_names and g != tf]
        if ko_cells >= min_ko_cells and len(present) >= min_module_genes:
            out[tf] = present
            print(f"  {tf}: {len(present)} target genes, {ko_cells} KO cells")
        else:
            print(f"  {tf}: SKIPPED (KO cells={ko_cells}, module genes found={len(present)})")
    if not out:
        print("  (none qualified)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default="data/eccite_filtered.h5ad")
    ap.add_argument("--panel", default="data/eccite_validation_panel.csv")
    ap.add_argument("--guide_col", default="perturbation",
                    help="obs column with the perturbation (falls back to guide_id)")
    ap.add_argument("--out", default="eccite_harness.h5ad")
    a = ap.parse_args()

    adata = ad.read_h5ad(a.h5ad)
    print(f"loaded {a.h5ad}: {adata.n_obs} cells x {adata.n_vars} genes")
    adata.X = densify(adata.X)

    # guide_target 
    col = a.guide_col if a.guide_col in adata.obs else "guide_id"
    raw = adata.obs[col].astype(str)
    adata.obs["guide_target"] = raw.map(strip_guide).values
    tg = pd.Series(adata.obs["guide_target"])
    targets = [t for t in tg.unique() if t != "NT"]
    var = set(adata.var_names)

    print(f"\nguide column used: '{col}'")
    print(f"controls (NT): {(tg=='NT').sum()} cells | perturbed: {(tg!='NT').sum()} cells")
    print("\nTest 1 feasibility (per target that is also a measured gene):")
    print(f"  {'target':10s} {'in_var':6s} {'KO_cells':8s} {'KO_zeros':8s} {'ctrl_zeros':10s}")
    X = adata.X
    name_to_col = {g: j for j, g in enumerate(adata.var_names)}
    ctrl = (tg == "NT").values
    usable = 0
    for t in sorted(targets):
        invar = t in var
        ko = (tg == t).values
        if invar:
            j = name_to_col[t]
            ko_zero = int(((X[:, j] == 0) & ko).sum())
            ctrl_zero = int(((X[:, j] == 0) & ctrl).sum())
            if ko_zero >= 10 and ctrl_zero >= 10:
                usable += 1
        else:
            ko_zero = ctrl_zero = 0
        print(f"  {t:10s} {str(invar):6s} {int(ko.sum()):8d} {ko_zero:8d} {ctrl_zero:10d}")
    print(f"  -> {usable} target(s) have enough zeros on both sides for Test 1")

    #  protein panel -> obsm 
    if os.path.exists(a.panel):
        panel = pd.read_csv(a.panel)
        print(f"\nprotein panel: {a.panel} | rows={len(panel)} | "
              f"proteins(genes)={sorted(panel['gene'].unique())}")
        pgenes = sorted(panel["gene"].unique())
        pnames = [f"prot_{g}" for g in pgenes]
        obs_index = pd.Index(adata.obs_names)
        prot = np.full((adata.n_obs, len(pgenes)), np.nan, dtype="float32")
        for k, g in enumerate(pgenes):
            sub = panel[panel["gene"] == g]
            pos = obs_index.get_indexer(sub["cell_id"].astype(str).values)
            ok = pos >= 0
            prot[pos[ok], k] = sub["measurement"].values[ok].astype("float32")
            covered = int(ok.sum())
            in_var = g in var
            print(f"  {g}: matched {covered}/{len(sub)} cells to data | "
                  f"gene in matrix: {in_var} | detected(=1): {int((sub['measurement']==1).sum())}")
        adata.obsm["protein"] = prot
        adata.uns["protein_names"] = pnames
        adata.uns["rna_for_protein"] = dict(zip(pnames, pgenes))

        # circularity check: is protein detection just tracking knockdown?
        # If the protein gene is ALSO the perturbed gene, knockdown lowers the
        # protein, so 'detected' may encode knockdown status -> Test 3 circular.
        print("\nTest 3 circularity check (protein detection vs knockdown status):")
        gt = pd.Series(adata.obs["guide_target"].values)
        for k, g in enumerate(pgenes):
            meas = prot[:, k]
            have = ~np.isnan(meas)
            is_ko = (gt == g).values & have
            is_ctrl = (gt == "NT").values & have
            det = meas > 0
            det_ko = det[is_ko].mean() if is_ko.sum() else float("nan")
            det_ctrl = det[is_ctrl].mean() if is_ctrl.sum() else float("nan")
            perturbed_here = g in set(targets)
            flag = "  <-- CONFOUNDED: protein gene is also perturbed" if perturbed_here else ""
            print(f"  {g}: detected in {det_ko:.2f} of {g}-KO cells vs "
                  f"{det_ctrl:.2f} of control cells{flag}")
            if perturbed_here:
                print(f"       -> on this file Test 3 for {g} mixes dropout with "
                      f"knockdown; use the full dataset where protein != perturbed gene.")
    else:
        print(f"\n!! panel {a.panel} not found -- Test 3 will be unavailable")

    adata.uns["tf_targets"] = build_tf_targets(targets, set(adata.var_names),
                                               tg, name_to_col, adata.X)

    # coerce indices and string/object columns to plain str so the h5ad write
    # doesn't trip on nullable-string arrays
    adata.obs.index = adata.obs.index.astype(str)
    adata.var.index = adata.var.index.astype(str)
    for c in adata.obs.columns:
        if adata.obs[c].dtype == object or str(adata.obs[c].dtype).startswith("string"):
            adata.obs[c] = adata.obs[c].astype(str)

    adata.write_h5ad(a.out)
    print(f"\nwrote {a.out}")

    # scan teacher outputs so we know what to score 
    print("\n===== teacher matrices on ECCITE (candidates to score) =====")
    stem = os.path.splitext(os.path.basename(a.h5ad))[0]
    seen = set()
    for method in ["scVI", "graph_smooth", "latent_truth", "MAGIC",
                   "svd_impute", "gene_median"]:
        for top in glob.glob(f"output/{method}/*{stem}*") + glob.glob(f"output/{method}/*eccite*"):
            if top in seen:
                continue
            seen.add(top)
            print(f"  [{top}]")
            for root, _dirs, files in os.walk(top):
                for f in sorted(files):
                    fp = os.path.join(root, f)
                    rel = os.path.relpath(fp, top)
                    if f.endswith((".npy", ".npz")):
                        try:
                            arr = np.load(fp, allow_pickle=True)
                            shp = getattr(arr, "shape", None)
                            tag = "  <== CELLS x GENES" if shp in [(adata.n_obs, adata.n_vars),
                                                                   (adata.n_vars, adata.n_obs)] else ""
                            print(f"     {rel}: shape={shp} dtype={getattr(arr,'dtype',None)}{tag}")
                        except Exception as e:
                            print(f"     {rel}: (load failed: {e})")
                    elif f.endswith((".h5ad", ".csv", ".json", ".pkl", ".pt")):
                        print(f"     {rel}: ({os.path.getsize(fp)//1024} KB)")


if __name__ == "__main__":
    main()
