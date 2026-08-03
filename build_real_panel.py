"""
build_real_panel.py
--------------------
Replaces make_mock.py. Builds eccite_validation_panel.csv from the REAL ADT
(antibody) surface-protein measurements in the Papalexi 2021 MuData object,
matched by verified cell barcode to the RNA h5ad used by the pipeline.

Key correctness safeguards:
  - Strips the lane prefix (l1_, l2_, ...) from MuData barcodes to align with
    the RNA file's barcodes.
  - DROPS any barcode that collides across lanes (appears in >1 lane), so no
    cell is ever assigned a protein value from the wrong physical cell.
  - Verifies guide-label agreement between the two files on matched cells and
    refuses to proceed if agreement isn't essentially perfect.
  - Writes CONTINUOUS protein values (not a 0/1 binarization). Thresholding is
    left to the validation step.

Output columns: cell_id, gene, measurement, assay
  where 'gene' is the RNA gene symbol the protein maps to:
    CD86 -> CD86, PDL1 -> CD274, PDL2 -> PDCD1LG2
  (CD366/TIM-3 has no relevant knockout target and is written but flagged.)

Run:
  python build_real_panel.py --mudata data/papalexi_2021.h5mu \
      --rna data/GSE153056_eccite.h5ad --out data/eccite_validation_panel_real.csv
"""
import argparse
import re
import collections
import numpy as np
import pandas as pd
import anndata as ad
import mudata as md
from scipy import sparse

# protein (ADT) name -> RNA gene symbol it corresponds to
PROT_TO_GENE = {
    "CD86": "CD86",
    "PDL1": "CD274",
    "PDL2": "PDCD1LG2",
    "CD366": "HAVCR2",   # TIM-3; included for completeness
}


def densify_col(X, j):
    col = X[:, j]
    if sparse.issparse(col):
        return np.asarray(col.todense()).ravel()
    return np.asarray(col).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mudata", default="data/papalexi_2021.h5mu")
    ap.add_argument("--rna", default="data/GSE153056_eccite.h5ad")
    ap.add_argument("--out", default="data/eccite_validation_panel_real.csv")
    ap.add_argument("--min_guide_agreement", type=float, default=0.999)
    a = ap.parse_args()

    print("Loading data...")
    m = md.read_h5mu(a.mudata)
    rna = ad.read_h5ad(a.rna)
    adt = m.mod["adt"]

    prot_names = list(adt.var_names)
    print(f"ADT proteins available: {prot_names}")

    # ---- build a collision-free barcode -> mudata row index map -------------
    stripped = [re.sub(r"^l\d+_", "", b) for b in adt.obs_names]
    counts = collections.Counter(stripped)
    keep = {k for k, v in counts.items() if v == 1}
    n_collide = sum(1 for v in counts.values() if v > 1)
    m_map = {re.sub(r"^l\d+_", "", b): i
             for i, b in enumerate(adt.obs_names)
             if re.sub(r"^l\d+_", "", b) in keep}
    print(f"Dropped {n_collide} barcodes that collide across lanes.")

    # ---- verify guide-label agreement before trusting the join -------------
    m_guide = (adt.obs["guide_ID"] if "guide_ID" in adt.obs
               else adt.obs["guide_id"]).values
    a_guide = rna.obs["guide_id"].astype(str).values
    agree = tot = 0
    matched_rows = []      # (rna_index, mudata_index, barcode)
    for ai, bc in enumerate(rna.obs_names):
        if bc in m_map:
            mi = m_map[bc]
            tot += 1
            if str(m_guide[mi]) == str(a_guide[ai]):
                agree += 1
            matched_rows.append((ai, mi, bc))
    frac = agree / tot if tot else 0.0
    print(f"Matched {tot} cells; guide-label agreement {agree}/{tot} = {frac:.4f}")
    if frac < a.min_guide_agreement:
        raise SystemExit(
            f"ABORT: guide agreement {frac:.4f} below "
            f"{a.min_guide_agreement}. Barcode join is not trustworthy.")

    # ---- ADT matrix ---------------------------------------------------------
    A = adt.X
    A = np.asarray(A.todense()) if sparse.issparse(A) else np.asarray(A)

    rna_var = {g: j for j, g in enumerate(rna.var_names)}

    # ---- emit one long-format row per (cell, protein) ----------------------
    rows = []
    for pj, pname in enumerate(prot_names):
        gene = PROT_TO_GENE.get(pname, pname)
        in_rna = gene in rna_var
        vals = A[[mi for (_, mi, _) in matched_rows], pj]
        for (ai, mi, bc), v in zip(matched_rows, vals):
            rows.append((bc, gene, float(v), "ADT", pname))
        print(f"  {pname} -> gene {gene} | in RNA matrix: {in_rna} | "
              f"cells: {len(vals)} | value range [{vals.min():.3f}, {vals.max():.3f}]")

    df = pd.DataFrame(rows, columns=["cell_id", "gene", "measurement", "assay", "protein"])
    df.to_csv(a.out, index=False)
    print(f"\nWrote {len(df)} rows to {a.out}")
    print("Values are CONTINUOUS ADT measurements (not binarized).")
    print("\nReminder for Test 3 design:")
    print("  - Use PDL1 (gene CD274) as the independent readout: measured, NOT a KO target.")
    print("  - CD86 and PDL2 (PDCD1LG2) are BOTH knockout targets -> confounded, do not use as independent readouts.")


if __name__ == "__main__":
    main()
