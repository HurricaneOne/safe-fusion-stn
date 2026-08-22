"""
atac_baseline_check.py
----------------------
Adds the one number the concept check was missing: the OVERALL RNA detection
rate across all mapped (gene, cell) pairs, ignoring ATAC entirely.

Why: the 13.4% (open) and 6.6% (closed) rates are only interpretable relative
to this baseline.
  - If baseline ~= 6.6%, the "closed" group is just average -> closed carries
    little information; the whole signal is in the open side being elevated.
  - If baseline is clearly above 6.6%, closed is genuinely suppressed -> the
    closed side is informative.

Uses the same strand-corrected promoter mapping as atac_concept_check_stranded.py.

Run:
  python atac_baseline_check.py
"""
import argparse
import numpy as np
import scanpy as sc
from scipy import sparse
from pyensembl import EnsemblRelease


def parse_interval(s):
    try:
        chrom, rng = s.split(":")
        a, b = rng.split("-")
        return chrom, int(a), int(b)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5")
    ap.add_argument("--promoter_window", type=int, default=2000)
    ap.add_argument("--ensembl_release", type=int, default=98)
    a = ap.parse_args()

    print("Loading multiome .h5 ...")
    adata = sc.read_10x_h5(a.h5, gex_only=False)
    adata.var_names_make_unique()
    is_rna = (adata.var["feature_types"] == "Gene Expression").values
    is_atac = (adata.var["feature_types"] == "Peaks").values
    rna = adata[:, is_rna]
    atac = adata[:, is_atac]

    # parse peaks
    peaks_by_chrom = {}
    for j, name in enumerate(atac.var_names):
        p = parse_interval(name)
        if p is None:
            continue
        chrom, s, e = p
        peaks_by_chrom.setdefault(chrom, ([], [], []))
        peaks_by_chrom[chrom][0].append(s)
        peaks_by_chrom[chrom][1].append(e)
        peaks_by_chrom[chrom][2].append(j)
    for chrom, (ss, ee, idx) in peaks_by_chrom.items():
        order = np.argsort(ss)
        peaks_by_chrom[chrom] = (np.array(ss)[order], np.array(ee)[order], np.array(idx)[order])

    # strand-aware promoter mapping
    ens = EnsemblRelease(a.ensembl_release)
    win = a.promoter_window
    gene_row, peak_col, mapped = [], [], []
    for gi, gid in enumerate(rna.var["gene_ids"].values):
        try:
            g = ens.gene_by_id(str(gid).split(".")[0])
        except Exception:
            continue
        chrom = "chr" + str(g.contig)
        if chrom not in peaks_by_chrom:
            continue
        tss = g.start if g.strand == "+" else g.end
        lo, hi = tss - win, tss + win
        starts, ends, idx = peaks_by_chrom[chrom]
        i1 = np.searchsorted(starts, hi, side="right")
        i0 = np.searchsorted(starts, lo - 5000, side="left")
        cand = np.arange(i0, i1)
        if len(cand) == 0:
            continue
        hit = idx[cand[ends[cand] >= lo]]
        if len(hit) == 0:
            continue
        mapped.append(gi)
        for pj in hit:
            gene_row.append(gi); peak_col.append(pj)

    gpm = sparse.csr_matrix((np.ones(len(gene_row)), (gene_row, peak_col)),
                            shape=(rna.n_vars, atac.n_vars))
    access_bool = (gpm @ atac.X.tocsr().T) > 0
    rna_det = (rna.X.tocsr().T > 0)

    mg = np.array(mapped)
    acc = access_bool[mg, :]
    det = rna_det[mg, :]

    total = acc.shape[0] * acc.shape[1]
    n_open = acc.nnz
    n_closed = total - n_open
    det_total = det.nnz                      # RNA detected anywhere in mapped set
    det_open = acc.multiply(det).nnz
    det_closed = det_total - det_open

    baseline = det_total / total
    rate_open = det_open / n_open
    rate_closed = det_closed / n_closed

    print("\n" + "=" * 58)
    print("BASELINE-CALIBRATED RESULT")
    print("=" * 58)
    print(f"  mapped genes: {len(mg)}   cells: {acc.shape[1]}")
    print(f"  total (gene,cell) pairs: {total:,}\n")
    print(f"  OVERALL RNA detection (baseline):   {baseline*100:5.2f}%")
    print(f"  RNA detection | promoter OPEN:      {rate_open*100:5.2f}%   "
          f"({(rate_open/baseline):.2f}x baseline)")
    print(f"  RNA detection | promoter CLOSED:    {rate_closed*100:5.2f}%   "
          f"({(rate_closed/baseline):.2f}x baseline)")
    print()
    # interpretation
    lift_open = rate_open / baseline
    drop_closed = rate_closed / baseline
    print("  Interpretation:")
    print(f"   - open is {lift_open:.2f}x baseline "
          f"({'elevated' if lift_open>1.1 else 'near baseline'})")
    print(f"   - closed is {drop_closed:.2f}x baseline "
          f"({'suppressed' if drop_closed<0.9 else 'near baseline'})")
    if drop_closed > 0.9:
        print("   >> CLOSED is close to baseline: the closed side adds little;")
        print("      the signal is mostly the OPEN side being elevated.")
    else:
        print("   >> CLOSED is meaningfully below baseline: it carries real")
        print("      true-zero information, not just average behavior.")
    print(f"\n  (fraction of pairs that are open: {n_open/total*100:.1f}%)")


if __name__ == "__main__":
    main()
