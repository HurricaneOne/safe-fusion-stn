"""
atac_concept_check.py
---------------------
Proof-of-concept for using chromatin accessibility (ATAC) as independent
evidence about RNA zeros.

Concept:
  For a gene in a cell, if the gene's PROMOTER has an accessible ATAC peak,
  the gene *can* be transcribed -> an RNA zero there is more likely a DROPOUT.
  If the promoter is inaccessible, the gene likely *cannot* be transcribed
  -> an RNA zero is more likely a TRUE zero.

Before building any imputer test, this script checks the premise:
  Does promoter accessibility actually predict RNA detection genome-wide?
  (Open promoter -> much higher RNA detection rate than closed promoter.)
If that relationship isn't strong, the concept is broken and we stop.

Data: 10x PBMC Multiome .h5 (RNA 'Gene Expression' + ATAC 'Peaks' in one file).

NOTE / limitation: the .h5 gives each gene's interval but not strand, so we
approximate the TSS by the interval's lower coordinate and use a +/-2kb promoter
window. This is an approximation; minus-strand genes have their true TSS at the
other end. Fine for a proof-of-concept; can be refined with a stranded annotation.

Run:
  python atac_concept_check.py \
      --h5 data/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5 \
      --promoter_window 2000
"""
import argparse
import numpy as np
import scanpy as sc
from scipy import sparse


def parse_interval(s):
    """'chr1:29553-30267' -> ('chr1', 29553, 30267); None if unparseable."""
    try:
        chrom, rng = s.split(":")
        a, b = rng.split("-")
        return chrom, int(a), int(b)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5")
    ap.add_argument("--promoter_window", type=int, default=2000,
                    help="+/- bp around the gene's approximate TSS")
    a = ap.parse_args()

    print("Loading multiome .h5 ...")
    adata = sc.read_10x_h5(a.h5, gex_only=False)
    adata.var_names_make_unique()

    is_rna = (adata.var["feature_types"] == "Gene Expression").values
    is_atac = (adata.var["feature_types"] == "Peaks").values
    print(f"cells={adata.n_obs}  RNA genes={is_rna.sum()}  ATAC peaks={is_atac.sum()}")

    rna = adata[:, is_rna]
    atac = adata[:, is_atac]

    # ---- parse peak coordinates -------------------------------------------
    peak_names = list(atac.var_names)
    peaks_by_chrom = {}   # chrom -> (starts[], ends[], local_peak_index[])
    for j, name in enumerate(peak_names):
        p = parse_interval(name)
        if p is None:
            continue
        chrom, s, e = p
        peaks_by_chrom.setdefault(chrom, ([], [], []))
        peaks_by_chrom[chrom][0].append(s)
        peaks_by_chrom[chrom][1].append(e)
        peaks_by_chrom[chrom][2].append(j)
    # sort each chrom's peaks by start
    for chrom, (ss, ee, idx) in peaks_by_chrom.items():
        order = np.argsort(ss)
        peaks_by_chrom[chrom] = (np.array(ss)[order],
                                 np.array(ee)[order],
                                 np.array(idx)[order])
    print(f"parsed peaks on {len(peaks_by_chrom)} chromosomes")

    # ---- map each gene to promoter peaks ----------------------------------
    win = a.promoter_window
    gene_intervals = rna.var["interval"].values
    gene_names = list(rna.var_names)

    gene_row, peak_col = [], []   # for building a (gene x peak) mapping matrix
    n_mapped = 0
    mapped_gene_local = []        # local RNA index of genes that got >=1 peak
    for gi, iv in enumerate(gene_intervals):
        p = parse_interval(str(iv))
        if p is None:
            continue
        chrom, gs, ge = p
        if chrom not in peaks_by_chrom:
            continue
        tss = gs                       # approx TSS = lower coord (see NOTE)
        lo, hi = tss - win, tss + win
        starts, ends, idx = peaks_by_chrom[chrom]
        # candidate peaks: start <= hi (and a little before), then filter end >= lo
        i1 = np.searchsorted(starts, hi, side="right")
        i0 = np.searchsorted(starts, lo - 5000, side="left")
        cand = np.arange(i0, i1)
        if len(cand) == 0:
            continue
        keep = ends[cand] >= lo
        hit = idx[cand[keep]]
        if len(hit) == 0:
            continue
        n_mapped += 1
        mapped_gene_local.append(gi)
        for pj in hit:
            gene_row.append(gi)
            peak_col.append(pj)

    print(f"genes with >=1 promoter peak: {n_mapped} of {len(gene_names)}")

    # ---- build per-cell promoter accessibility ----------------------------
    # gene_peak_map: (n_genes x n_peaks) binary
    n_genes = rna.n_vars
    n_peaks = atac.n_vars
    gpm = sparse.csr_matrix(
        (np.ones(len(gene_row)), (gene_row, peak_col)),
        shape=(n_genes, n_peaks))

    # accessibility[gene, cell] = sum of that gene's promoter peak counts in cell
    # atac.X is cells x peaks; we want genes x cells
    atacX = atac.X.tocsr()                    # cells x peaks
    access = gpm @ atacX.T                     # genes x cells (sparse)
    access_bool = (access > 0)                 # promoter open?

    # RNA detection for the same genes
    rnaX = rna.X.tocsr()                       # cells x genes
    rna_det = (rnaX.T > 0)                      # genes x cells boolean

    # restrict to mapped genes only (rows with a promoter peak)
    mg = np.array(mapped_gene_local)
    acc = access_bool[mg, :]
    det = rna_det[mg, :]

    # ---- SANITY CHECK: does open promoter -> higher RNA detection? ---------
    total_entries = acc.shape[0] * acc.shape[1]
    n_open = acc.nnz
    n_closed = total_entries - n_open
    det_and_open = acc.multiply(det).nnz
    total_det = det.nnz
    det_and_closed = total_det - det_and_open

    rate_open = det_and_open / n_open if n_open else float("nan")
    rate_closed = det_and_closed / n_closed if n_closed else float("nan")

    print("\n" + "=" * 60)
    print("SANITY CHECK: promoter accessibility vs RNA detection")
    print("=" * 60)
    print(f"  (gene,cell) entries analyzed: {total_entries:,} "
          f"(mapped genes x cells)")
    print(f"  promoter OPEN entries:   {n_open:,}")
    print(f"  promoter CLOSED entries: {n_closed:,}")
    print(f"\n  RNA detected when promoter OPEN:   {rate_open:.4f}")
    print(f"  RNA detected when promoter CLOSED: {rate_closed:.4f}")
    if rate_closed > 0:
        print(f"  ratio (open / closed): {rate_open / rate_closed:.2f}x")
    print()
    if rate_open > 2 * rate_closed:
        print("  >> CONCEPT HOLDS: open promoters have much higher RNA detection.")
        print("     ATAC accessibility carries real information about whether a")
        print("     zero is a dropout vs a true zero. Safe to build the test.")
    else:
        print("  >> WEAK: accessibility does not strongly predict detection here.")
        print("     Investigate before building the test (promoter window, TSS")
        print("     strand approximation, or QC may be the issue).")


if __name__ == "__main__":
    main()
