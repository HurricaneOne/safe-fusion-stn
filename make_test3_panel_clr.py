"""
make_test3_panel_clr.py
-----------------------
CLR (centered log-ratio) normalization of the real ADT panel, then extract
PDL1 (gene CD274) as the independent Test 3 readout.

CLR is computed PER CELL across that cell's proteins (Seurat margin=2):
    clr(x_i) = ln( (1 + x_i) / geomean_j(1 + x_j) )
where the geometric mean runs over all proteins measured in that cell.
This is why we normalize the FULL 4-protein panel before extracting PDL1 --
the other proteins define the per-cell baseline.

After CLR, "detectable" is thresholded. Because CLR is centered, a natural
cutoff is 0 (protein above the cell's own geometric-mean baseline), but we
default to a quantile of the CLR values for comparability with the quick pass.

Run:
  python make_test3_panel_clr.py --panel data/eccite_validation_panel_real.csv \
      --out data/eccite_panel_pdl1_clr.csv --protein PDL1 --gene CD274 \
      --threshold quantile --quantile 0.5 --binarize
"""
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/eccite_validation_panel_real.csv")
    ap.add_argument("--out", default="data/eccite_panel_pdl1_clr.csv")
    ap.add_argument("--protein", default="PDL1")
    ap.add_argument("--gene", default="CD274")
    ap.add_argument("--threshold", choices=["quantile", "zero"], default="quantile",
                    help="'zero': detectable if CLR>0 (above cell baseline); "
                         "'quantile': detectable if CLR> that quantile")
    ap.add_argument("--quantile", type=float, default=0.5)
    ap.add_argument("--binarize", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.panel)

    # pivot to a cells x proteins matrix (rows=cell, cols=protein)
    wide = df.pivot_table(index="cell_id", columns="protein",
                          values="measurement", aggfunc="first")
    proteins = list(wide.columns)
    print(f"Proteins used for per-cell CLR baseline: {proteins}")
    X = wide.values.astype("float64")   # cells x proteins

    # ---- CLR per cell (across proteins) ------------------------------------
    log1p = np.log1p(X)                                  # ln(1 + x)
    # geometric mean of (1+x) per cell = exp(mean of ln(1+x) across proteins)
    gm_log = log1p.mean(axis=1, keepdims=True)           # ln(geomean) per cell
    clr = log1p - gm_log                                 # ln((1+x)/geomean)

    clr_df = pd.DataFrame(clr, index=wide.index, columns=proteins)

    if a.protein not in clr_df.columns:
        raise SystemExit(f"{a.protein} not in panel. Have: {proteins}")
    vals = clr_df[a.protein]

    # ---- threshold ---------------------------------------------------------
    if a.threshold == "zero":
        thr = 0.0
    else:
        thr = float(np.quantile(vals.values, a.quantile))

    detect = (vals.values > thr)
    print(f"\n{a.protein} -> gene {a.gene} (CLR-normalized)")
    print(f"  cells: {len(vals)}")
    print(f"  CLR range: [{vals.min():.3f}, {vals.max():.3f}]")
    print(f"  threshold ({a.threshold}): {thr:.3f}")
    print(f"  detectable: {int(detect.sum())} ({100*detect.mean():.1f}%)")

    out = pd.DataFrame({"cell_id": vals.index, "gene": a.gene})
    if a.binarize:
        out["measurement"] = detect.astype(int)
        print("  wrote BINARY detection flag (0/1)")
    else:
        out["measurement"] = vals.values
        print("  wrote CONTINUOUS CLR values")

    out.to_csv(a.out, index=False)
    print(f"\nWrote {len(out)} rows to {a.out}")


if __name__ == "__main__":
    main()
