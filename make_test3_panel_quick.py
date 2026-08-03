"""
make_test3_panel_quick.py
-------------------------
Quick version of the Test 3 panel: take the REAL continuous ADT panel and keep
only PDL1 (gene CD274) -- the independent readout that was measured but NOT a
knockout target. Writes it in the format build_eccite_harness.py expects
(cell_id, gene, measurement), with a quantile-based binary detection flag so
the existing --binary_panel path works.

This is the QUICK pass: raw ADT, quantile threshold, no CLR normalization yet.

Run:
  python make_test3_panel_quick.py --panel data/eccite_validation_panel_real.csv \
      --out data/eccite_panel_pdl1_quick.csv --quantile 0.5
"""
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/eccite_validation_panel_real.csv")
    ap.add_argument("--out", default="data/eccite_panel_pdl1_quick.csv")
    ap.add_argument("--protein", default="PDL1", help="ADT protein name to use")
    ap.add_argument("--gene", default="CD274", help="RNA gene it maps to")
    ap.add_argument("--quantile", type=float, default=0.5,
                    help="detectable if raw ADT > this quantile of the protein")
    ap.add_argument("--binarize", action="store_true",
                    help="write 0/1 detection flag (for --binary_panel); "
                         "otherwise keep continuous values")
    a = ap.parse_args()

    df = pd.read_csv(a.panel)

    # keep only the chosen independent readout
    sub = df[df["protein"] == a.protein].copy()
    if len(sub) == 0:
        raise SystemExit(f"No rows for protein {a.protein}. "
                         f"Available: {sorted(df['protein'].unique())}")

    thr = np.quantile(sub["measurement"].values, a.quantile)
    print(f"{a.protein} -> gene {a.gene}")
    print(f"  cells: {len(sub)}")
    print(f"  raw ADT range: [{sub['measurement'].min():.1f}, {sub['measurement'].max():.1f}]")
    print(f"  quantile {a.quantile} threshold: {thr:.2f}")
    print(f"  detectable (> thr): {int((sub['measurement'] > thr).sum())} "
          f"({100*(sub['measurement'] > thr).mean():.1f}%)")

    out = pd.DataFrame({"cell_id": sub["cell_id"].values, "gene": a.gene})
    if a.binarize:
        out["measurement"] = (sub["measurement"].values > thr).astype(int)
        print("  wrote BINARY detection flag (0/1)")
    else:
        out["measurement"] = sub["measurement"].values
        print("  wrote CONTINUOUS values")

    out.to_csv(a.out, index=False)
    print(f"\nWrote {len(out)} rows to {a.out}")


if __name__ == "__main__":
    main()
