"""
atac_concept_check_stranded.py
------------------------------
Strand-corrected version of atac_concept_check.py.

The original approximated each gene's TSS by the lower coordinate of its
interval, which is wrong for minus-strand genes (their TSS is at the HIGH
coordinate). Here we look up the real strand and TSS from pyensembl and place
the promoter window on the correct (upstream) side:
    + strand: TSS = gene.start,  promoter = [TSS - win, TSS + win]
    - strand: TSS = gene.end,    promoter = [TSS - win, TSS + win]

Everything else (peak parsing, mapping, sanity check) matches the original so
the ratios are directly comparable.

Run:
  python atac_concept_check_stranded.py \
      --h5 data/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5 \
      --promoter_window 2000 --ensembl_release 98
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
    print(f"cells={adata.n_obs}  RNA genes={is_rna.sum()}  ATAC peaks={is_atac.sum()}")

    # ---- parse peaks -------------------------------------------------------
    peak_names = list(atac.var_names)
    peaks_by_chrom = {}
    for j, name in enumerate(peak_names):
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

    # ---- strand-aware TSS from pyensembl ----------------------------------
    print("Looking up strand/TSS from pyensembl ...")
    ens = EnsemblRelease(a.ensembl_release)
    gene_ids = rna.var["gene_ids"].values           # ENSG... ids
    win = a.promoter_window

    gene_row, peak_col = [], []
    mapped_gene_local = []
    n_lookup_fail = 0
    for gi, gid in enumerate(gene_ids):
        # strip version suffix if present (ENSG000...".3")
        gid_clean = str(gid).split(".")[0]
        try:
            g = ens.gene_by_id(gid_clean)
        except Exception:
            n_lookup_fail += 1
            continue
        chrom = "chr" + str(g.contig)               # pyensembl uses '1', peaks use 'chr1'
        if chrom not in peaks_by_chrom:
            continue
        # strand-correct TSS
        tss = g.start if g.strand == "+" else g.end
        lo, hi = tss - win, tss + win
        starts, ends, idx = peaks_by_chrom[chrom]
        i1 = np.searchsorted(starts, hi, side="right")
        i0 = np.searchsorted(starts, lo - 5000, side="left")
        cand = np.arange(i0, i1)
        if len(cand) == 0:
            continue
        keep = ends[cand] >= lo
        hit = idx[cand[keep]]
        if len(hit) == 0:
            continue
        mapped_gene_local.append(gi)
        for pj in hit:
            gene_row.append(gi)
            peak_col.append(pj)

    print(f"pyensembl lookup failures: {n_lookup_fail}")
    print(f"genes with >=1 promoter peak: {len(mapped_gene_local)} of {rna.n_vars}")

    # ---- accessibility + sanity check (identical to original) -------------
    gpm = sparse.csr_matrix((np.ones(len(gene_row)), (gene_row, peak_col)),
                            shape=(rna.n_vars, atac.n_vars))
    atacX = atac.X.tocsr()
    access_bool = (gpm @ atacX.T) > 0
    rna_det = (rna.X.tocsr().T > 0)

    mg = np.array(mapped_gene_local)
    acc = access_bool[mg, :]
    det = rna_det[mg, :]

    total_entries = acc.shape[0] * acc.shape[1]
    n_open = acc.nnz
    n_closed = total_entries - n_open
    det_and_open = acc.multiply(det).nnz
    total_det = det.nnz
    det_and_closed = total_det - det_and_open
    rate_open = det_and_open / n_open if n_open else float("nan")
    rate_closed = det_and_closed / n_closed if n_closed else float("nan")

    print("\n" + "=" * 60)
    print("SANITY CHECK (strand-corrected): promoter access vs RNA detection")
    print("=" * 60)
    print(f"  entries analyzed: {total_entries:,}")
    print(f"  promoter OPEN:   {n_open:,}")
    print(f"  promoter CLOSED: {n_closed:,}")
    print(f"\n  RNA detected when promoter OPEN:   {rate_open:.4f}")
    print(f"  RNA detected when promoter CLOSED: {rate_closed:.4f}")
    if rate_closed > 0:
        print(f"  ratio (open / closed): {rate_open / rate_closed:.2f}x")
    print("\n  (compare to the un-stranded run's 2.04x)")


if __name__ == "__main__":
    main()
