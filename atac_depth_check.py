"""
atac_depth_check.py
-------------------
Depth-confounder check (simplest stratified version).

Worry: cells sequenced deeply on both RNA and ATAC show more open promoters AND
more RNA detection, which can manufacture an open-vs-detected association with
no real biological discrimination.

Test: bin cells by RNA depth and ATAC depth, then recompute the open-vs-closed
RNA detection ratio WITHIN each bin (depth held roughly constant). If the ratio
stays ~2x within bins, depth is not the driver and the signal is real. If it
collapses toward 1x, the pooled effect was mostly depth.

Run:
  python atac_depth_check.py --n_bins 5
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
    ap.add_argument("--n_bins", type=int, default=5, help="quantile bins per depth axis")
    ap.add_argument("--min_cells", type=int, default=50, help="skip strata smaller than this")
    a = ap.parse_args()

    print("Loading multiome .h5 ...")
    adata = sc.read_10x_h5(a.h5, gex_only=False)
    adata.var_names_make_unique()
    is_rna = (adata.var["feature_types"] == "Gene Expression").values
    is_atac = (adata.var["feature_types"] == "Peaks").values
    rna = adata[:, is_rna]
    atac = adata[:, is_atac]

    # ---- per-cell depths (technical sampling) -----------------------------
    rna_depth = np.asarray(rna.X.sum(axis=1)).ravel()    # total UMIs per cell
    atac_depth = np.asarray(atac.X.sum(axis=1)).ravel()  # total fragments per cell
    print(f"RNA depth: median={np.median(rna_depth):.0f} "
          f"[{rna_depth.min():.0f}, {rna_depth.max():.0f}]")
    print(f"ATAC depth: median={np.median(atac_depth):.0f} "
          f"[{atac_depth.min():.0f}, {atac_depth.max():.0f}]")

    # ---- promoter mapping (same as before) --------------------------------
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
    # genes x cells, restricted to mapped genes, as CSC for fast column slicing
    mg = np.array(mapped)
    acc = ((gpm @ atac.X.tocsr().T) > 0).tocsc()[mg, :]
    det = (rna.X.tocsr().T > 0).tocsc()[mg, :]

    # pooled reference
    pooled_open = acc.nnz
    pooled_det_open = acc.multiply(det).nnz
    pooled_closed = acc.shape[0]*acc.shape[1] - pooled_open
    pooled_det_closed = det.nnz - pooled_det_open
    pooled_rr = (pooled_det_open/pooled_open) / (pooled_det_closed/pooled_closed)
    print(f"\nPOOLED open/closed relative risk (no depth control): {pooled_rr:.2f}x")

    # ---- bin cells by both depths -----------------------------------------
    def qbin(x, k):
        edges = np.quantile(x, np.linspace(0, 1, k+1))
        edges[-1] += 1e-9
        return np.clip(np.digitize(x, edges[1:-1]), 0, k-1)

    rb = qbin(rna_depth, a.n_bins)
    ab = qbin(atac_depth, a.n_bins)

    print(f"\nWithin-depth-stratum open/closed relative risk "
          f"({a.n_bins}x{a.n_bins} bins):")
    print(f"  {'rna_bin':>8s} {'atac_bin':>8s} {'cells':>6s} "
          f"{'open%RNA':>9s} {'closed%RNA':>10s} {'ratio':>7s}")
    ratios, weights = [], []
    for ri in range(a.n_bins):
        for ai in range(a.n_bins):
            cells = np.where((rb == ri) & (ab == ai))[0]
            if len(cells) < a.min_cells:
                continue
            A = acc[:, cells]; D = det[:, cells]
            n_open = A.nnz
            n_closed = A.shape[0]*A.shape[1] - n_open
            d_open = A.multiply(D).nnz
            d_closed = D.nnz - d_open
            if n_open == 0 or n_closed == 0 or d_closed == 0:
                continue
            r_open = d_open/n_open
            r_closed = d_closed/n_closed
            rr = r_open/r_closed
            ratios.append(rr); weights.append(len(cells))
            print(f"  {ri:8d} {ai:8d} {len(cells):6d} "
                  f"{r_open*100:8.2f}% {r_closed*100:9.2f}% {rr:6.2f}x")

    ratios = np.array(ratios); weights = np.array(weights)
    if len(ratios):
        wmean = np.average(ratios, weights=weights)
        print(f"\n  strata reported: {len(ratios)}")
        print(f"  within-stratum ratio: median={np.median(ratios):.2f}x  "
              f"cell-weighted mean={wmean:.2f}x  "
              f"range=[{ratios.min():.2f}, {ratios.max():.2f}]")
        print(f"  pooled (uncontrolled): {pooled_rr:.2f}x")
        print()
        if wmean > 1.4:
            print("  >> SURVIVES: the open/closed gap persists within depth strata.")
            print("     Depth is not the main driver; the signal looks real.")
        elif wmean > 1.15:
            print("  >> PARTIAL: the gap shrinks within strata but doesn't vanish.")
            print("     Depth explains some of it; residual signal remains.")
        else:
            print("  >> COLLAPSES: within equal-depth cells the gap nearly disappears.")
            print("     The pooled effect was largely a depth artifact.")


if __name__ == "__main__":
    main()
