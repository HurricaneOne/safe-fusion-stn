"""
atac_within_gene_check.py
-------------------------
Within-gene confounder check.

Worry: pooling all ~21,000 genes, the open-vs-detected association could come
from BETWEEN-gene differences (housekeeping genes are open+detected everywhere;
silent genes are closed+undetected everywhere) rather than from ATAC
discriminating cell states WITHIN a gene -- which is what zero-classification
actually needs.

Test: for each gene, using only that gene's own cells, compare RNA detection in
open-promoter vs closed-promoter cells. Summarize the within-gene effect across
genes. If the effect persists within genes, ATAC discriminates at the level the
task requires. If it collapses to ~1x, the pooled signal was between-gene.

Only genes with enough BOTH open and closed cells are testable (a gene that is
open in every cell, or closed in every cell, poses no within-gene comparison).

Run:
  python atac_within_gene_check.py --min_open 20 --min_closed 20
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
    ap.add_argument("--min_open", type=int, default=20,
                    help="min open cells for a gene to be testable")
    ap.add_argument("--min_closed", type=int, default=20,
                    help="min closed cells for a gene to be testable")
    a = ap.parse_args()

    print("Loading multiome .h5 ...")
    adata = sc.read_10x_h5(a.h5, gex_only=False)
    adata.var_names_make_unique()
    is_rna = (adata.var["feature_types"] == "Gene Expression").values
    is_atac = (adata.var["feature_types"] == "Peaks").values
    rna = adata[:, is_rna]
    atac = adata[:, is_atac]

    # promoter mapping (same as before)
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
    mg = np.array(mapped)
    # genes x cells, as CSR for fast row (per-gene) access
    acc = ((gpm @ atac.X.tocsr().T) > 0).tocsr()[mg, :]
    det = (rna.X.tocsr().T > 0).tocsr()[mg, :]

    n_genes, n_cells = acc.shape
    print(f"mapped genes: {n_genes}   cells: {n_cells}")

    # ---- per-gene open/closed detection -----------------------------------
    # For each gene row: n_open, det_open, n_closed, det_closed
    open_per_gene = np.asarray(acc.sum(axis=1)).ravel()            # cells open
    det_open_per_gene = np.asarray(acc.multiply(det).sum(axis=1)).ravel()
    det_total_per_gene = np.asarray(det.sum(axis=1)).ravel()
    closed_per_gene = n_cells - open_per_gene
    det_closed_per_gene = det_total_per_gene - det_open_per_gene

    # testable genes: enough open AND closed cells
    testable = (open_per_gene >= a.min_open) & (closed_per_gene >= a.min_closed)
    nt = int(testable.sum())
    print(f"testable genes (>= {a.min_open} open & >= {a.min_closed} closed cells): {nt}")

    ro = det_open_per_gene[testable] / open_per_gene[testable]
    rc = det_closed_per_gene[testable] / closed_per_gene[testable]

    # within-gene ratio where closed rate > 0 (avoid div by zero)
    ok = rc > 0
    within_ratio = ro[ok] / rc[ok]
    risk_diff = ro - rc            # percentage-point difference, defined for all

    # ---- gene-stratified pooled relative risk -----------------------------
    # Hold gene identity constant: sum within-gene counts, but only over
    # testable genes, then form the pooled rate ratio.
    tot_det_open = det_open_per_gene[testable].sum()
    tot_open = open_per_gene[testable].sum()
    tot_det_closed = det_closed_per_gene[testable].sum()
    tot_closed = closed_per_gene[testable].sum()
    strat_rr = (tot_det_open/tot_open) / (tot_det_closed/tot_closed)

    print("\n" + "=" * 58)
    print("WITHIN-GENE RESULT")
    print("=" * 58)
    print(f"  median within-gene ratio (open/closed): {np.median(within_ratio):.2f}x")
    print(f"  mean within-gene ratio:                 {np.mean(within_ratio):.2f}x")
    print(f"  fraction of genes with ratio > 1:       {np.mean(within_ratio > 1)*100:.1f}%")
    print(f"  median within-gene risk difference:     "
          f"{np.median(risk_diff)*100:.2f} pct points")
    print(f"\n  gene-stratified pooled relative risk:   {strat_rr:.2f}x")
    print(f"  (compare to the uncontrolled pooled ~2.03x)")
    print()
    m = np.median(within_ratio)
    if m > 1.4:
        print("  >> SURVIVES: the effect holds WITHIN genes. ATAC discriminates")
        print("     cell states for the same gene -- useful for zero classification.")
    elif m > 1.15:
        print("  >> PARTIAL: within-gene effect is weaker but present. Some of the")
        print("     pooled signal was between-gene, but a real within-gene effect remains.")
    else:
        print("  >> COLLAPSES: within genes the effect nearly disappears. The pooled")
        print("     signal was mostly between-gene (housekeeping vs silent), which does")
        print("     NOT help classify an individual gene's zeros.")


if __name__ == "__main__":
    main()
