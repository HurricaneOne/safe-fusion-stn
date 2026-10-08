#!/usr/bin/env python3
"""Multi-gene cross-modal protein validation on the Hao 2021 CITE-seq units.

This generalizes ``scripts/sle_evaluate_protein.py`` (single protein, FcRL3) to
EVERY antibody tag that maps cleanly to an RNA gene, and adds a *pre-registered*
gene-selection gate so the tested gene list is fixed from observed RNA-vs-protein
agreement BEFORE any imputation method is scored.

Logic, in the mentor's words:
  1. Pick the genes first. Keep only genes whose RNA and protein clearly go
     together in cells where RNA IS observed (counts > 0) and protein is
     measured, at least as clearly as for CD274. This uses NO imputed value, so
     it cannot favor any method. The list is written to selected_genes.csv and
     frozen before stage 2.
  2. For each kept gene, take the cells where the RNA read is a recorded zero
     (and that are held out -- test donors in each fold).
  3. Let each method (Safe Fusion, scVI, SVD, weighted kNN, MAGIC) rank those
     cells by its imputed value ("most likely really on" -> "most likely off").
  4. Score each ranking against the measured protein: donor-weighted Spearman
     between the method's score and the CLR protein, plus mean threshold AUROC
     over CLR quantiles 0.30-0.70 (identical metric to the FcRL3 evaluator).
  5. Count the genes where Safe Fusion ranks best.

Everything reuses the already-computed ``cite`` deploy fills, so this is a cheap
CPU-only pass -- no re-imputation.

NOTE on gene panels: each fold's truth.h5ad has its OWN gene panel (same ~43
fixed/focus genes, but a different top-1200 "variable genes" selection, since
that selection is computed from each fold's own development-donor subset).
Genes are therefore matched and pooled across folds by SYMBOL, never by a
shared integer index -- a kept gene may legitimately be present in some folds
and absent from others, which is reported per gene (n_folds_present), not
treated as an error.

Usage (run select first, eyeball the two CSVs, then evaluate):
  python scripts/sle_evaluate_protein_multigene.py --stage select
  #   -> review artifacts/.../q7_protein_multigene/tag_gene_map.csv
  #   -> review artifacts/.../q7_protein_multigene/selected_genes.csv
  python scripts/sle_evaluate_protein_multigene.py --stage evaluate
  # or do both at once (the frozen list is still written before any scoring):
  python scripts/sle_evaluate_protein_multigene.py --stage all
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

from sle_common import (
    CASE,
    FOLDS,
    METHODS,
    SEED,
    bootstrap_draws,
    interval,
    load_fills,
    load_unit,
    pair_auroc,
    weighted_auroc,
)

QUANTILES = np.linspace(0.30, 0.70, 41)
SET_NAME = "cite"

# ---------------------------------------------------------------------------
# Antibody tag -> RNA gene symbol.
#
# The ADT tags in GSE164378 are named by their CD/clone label (e.g.
# "CD307c/FcRL3"), which is often NOT the HGNC gene symbol. CURATED lists the
# unambiguous CD-label -> gene mappings where they differ. AUTO-matching then
# handles tags whose label already is (or contains) the symbol. Anything that
# is ambiguous (one antibody, several candidate genes), an isoform-specific
# reagent that RNA cannot distinguish, or an isotype control is DROPPED, with a
# reason recorded in tag_gene_map.csv for human review. Nothing enters the test
# that is not a clean one-protein / one-gene measurement.
# ---------------------------------------------------------------------------
CURATED: dict[str, str] = {
    "CD11a": "ITGAL", "CD11b": "ITGAM", "CD11c": "ITGAX", "CD16": "FCGR3A",
    "CD32": "FCGR2A", "CD56": "NCAM1", "CD57": "B3GAT1", "CD62L": "SELL",
    "CD62P": "SELP", "CD66b": "CEACAM8", "CD103": "ITGAE", "CD122": "IL2RB",
    "CD123": "IL3RA", "CD124": "IL4R", "CD127": "IL7R", "CD137": "TNFRSF9",
    "CD134": "TNFRSF4", "CD152": "CTLA4", "CD154": "CD40LG", "CD183": "CXCR3",
    "CD184": "CXCR4", "CD185": "CXCR5", "CD186": "CXCR6", "CD194": "CCR4",
    "CD195": "CCR5", "CD196": "CCR6", "CD197": "CCR7", "CD223": "LAG3",
    "CD226": "CD226", "CD244": "CD244", "CD267": "TNFRSF13B", "CD268": "TNFRSF13C",
    "CD269": "TNFRSF17", "CD272": "BTLA", "CD273": "PDCD1LG2", "CD274": "CD274",
    "CD275": "ICOSLG", "CD278": "ICOS", "CD279": "PDCD1", "CD303": "CLEC4C",
    "CD304": "NRP1", "CD307c": "FCRL3", "CD314": "KLRK1", "CD319": "SLAMF7",
    "CD328": "SIGLEC7", "CD335": "NCR1", "CD337": "NCR3", "CD352": "SLAMF6",
    "CD354": "TREM1", "CD355": "CRTAM", "CD366": "HAVCR2", "CD64": "FCGR1A",
    "CD71": "TFRC", "CD73": "NT5E", "CD39": "ENTPD1", "CD35": "CR1",
    "CD88": "C5AR1", "CD94": "KLRD1", "CD95": "FAS", "CD98": "SLC3A2",
    "CD112": "NECTIN2", "CD141": "THBD", "CD155": "PVR", "CD158e1": "KIR3DL1",
    "CD161": "KLRB1", "CD169": "SIGLEC1", "CD172a": "SIRPA", "CD85j": "LILRB1",
    "CD200": "CD200", "CD201": "PROCR", "CD254": "TNFSF11", "CD29": "ITGB1",
    "CD41": "ITGA2B", "CD49a": "ITGA1", "CD49b": "ITGA2", "CD49d": "ITGA4",
    "CD49f": "ITGA6", "CD54": "ICAM1", "CD61": "ITGB3", "CD25": "IL2RA",
    "CD30": "TNFRSF8", "CD1c": "CD1C", "CD1d": "CD1D", "FceRIa": "FCER1A",
    "IgD": "IGHD", "IgM": "IGHM", "KLRG1": "KLRG1", "TIGIT": "TIGIT",
    "CLEC12A": "CLEC12A", "FOLR2": "FOLR2", "CX3CR1": "CX3CR1", "CD45": "PTPRC",
}
# Tags that look mappable but are intentionally excluded. Keys are CD/clone
# labels; value is the reason shown in the map table.
DROP_LABELS: dict[str, str] = {
    "CD3": "pan-CD3: RNA cannot be assigned to CD3D/CD3E/CD3G",
    "CD8": "pan-CD8: ambiguous between CD8A/CD8B (use CD8a tag if present)",
    "CD45RA": "PTPRC isoform-specific; RNA cannot distinguish RA/RO",
    "CD45RO": "PTPRC isoform-specific; RNA cannot distinguish RA/RO",
    "TCR": "TCR reagent: no single RNA gene",
    "HLA-DR": "multiple HLA-DR genes; not a single symbol",
    "HLA-ABC": "multiple HLA class I genes; not a single symbol",
    "HLA-E": "keep only if a single HLA-E gene is intended -- review",
    "Ig-light-chain": "kappa/lambda: no single RNA gene",
}
ISOTYPE_PATTERNS = ("IgG1", "IgG2a", "IgG2b", "IgG2c", "IgG1-", "Isotype",
                    "isotype", "Rat-Ig", "Mouse-Ig", "Armenian", "Hamster",
                    "Control", "control")


def read_tags(adt_path: Path) -> list[str]:
    """Return the antibody tags present in the adt.parquet (from clr| columns)."""
    columns = pq.ParquetFile(adt_path).schema.names
    return [name[len("clr|"):] for name in columns if name.startswith("clr|")]


def _candidate_symbols(tag: str) -> list[str]:
    """Token candidates from a tag, e.g. 'CD307c/FcRL3' -> ['CD307C','FCRL3']."""
    out: list[str] = []
    for part in re.split(r"[/]", tag):
        part = part.strip()
        if not part:
            continue
        out.append(part.upper())
        stripped = re.match(r"^(.*?)(?:[-_]\d+)$", part)  # drop lane suffix 'CD3-1'
        if stripped:
            out.append(stripped.group(1).upper())
    # de-duplicate, preserve order
    seen, unique = set(), []
    for sym in out:
        if sym not in seen:
            seen.add(sym)
            unique.append(sym)
    return unique


def map_tag_to_gene(tag: str, gene_set: set[str]) -> tuple[str | None, str]:
    """Map one tag to a gene present in the RNA panel (any fold), or explain why not."""
    if any(pattern in tag for pattern in ISOTYPE_PATTERNS):
        return None, "isotype/control"
    for label, reason in DROP_LABELS.items():
        if tag == label or tag.split("/")[0] == label:
            return None, f"dropped: {reason}"
    # curated label match (on the whole tag or its first slash component)
    for key in (tag, tag.split("/")[0]):
        if key in CURATED:
            gene = CURATED[key]
            return (gene, "curated") if gene in gene_set else (None, f"curated->{gene} absent from RNA")
    # auto: a token that is itself a gene symbol present in the panel
    hits = [sym for sym in _candidate_symbols(tag) if sym in gene_set]
    if len(hits) == 1:
        return hits[0], "auto-token"
    if len(hits) > 1:
        return None, f"ambiguous tokens {hits}"
    return None, "no RNA gene match"


def build_tag_gene_map(adt_path: Path, gene_set: set[str]) -> pd.DataFrame:
    rows = []
    for tag in read_tags(adt_path):
        gene, reason = map_tag_to_gene(tag, gene_set)
        rows.append({"tag": tag, "gene": gene, "status": "kept" if gene else "dropped", "reason": reason})
    frame = pd.DataFrame(rows)
    # one gene per test: if several kept tags map to the same gene, keep the
    # first and mark the rest for review rather than testing the gene twice.
    kept = frame["gene"].notna()
    dup = kept & frame["gene"].duplicated(keep="first")
    frame.loc[dup, ["status", "reason"]] = ["dropped", "duplicate antibody for this gene"]
    frame.loc[dup, "gene_kept_elsewhere"] = frame.loc[dup, "gene"]
    frame.loc[dup, "gene"] = None
    return frame


# --------------------------- metrics (as in FcRL3) --------------------------
def weighted_spearman(x: np.ndarray, y: np.ndarray, item_weights: np.ndarray) -> np.ndarray:
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    total = item_weights.sum(axis=1, keepdims=True)
    mx = item_weights @ rx / total[:, 0]
    my = item_weights @ ry / total[:, 0]
    cov = item_weights @ (rx * ry) / total[:, 0] - mx * my
    vx = item_weights @ (rx * rx) / total[:, 0] - mx**2
    vy = item_weights @ (ry * ry) / total[:, 0] - my**2
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.sqrt(vx * vy)


def threshold_auroc(score: np.ndarray, protein: np.ndarray, donor_index: np.ndarray,
                    donors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    aurocs = []
    for quantile in QUANTILES:
        high = protein > np.quantile(protein, quantile)
        pos = [score[high & (donor_index == d)] for d in range(len(donors))]
        neg = [score[~high & (donor_index == d)] for d in range(len(donors))]
        n_pos = np.asarray([len(v) for v in pos], float)
        n_neg = np.asarray([len(v) for v in neg], float)
        aurocs.append(weighted_auroc(pair_auroc(pos, neg), n_pos, n_neg, weights, weights))
    return np.nanmean(aurocs, axis=0)


# --------------------------- data assembly ----------------------------------
def read_union_genes() -> tuple[set[str], dict[str, int]]:
    """Union of gene symbols across ALL folds' truth.h5ad (read backed, cheap).
    Used only to decide which antibodies are eligible to map to a gene at all;
    a gene need not be in every fold to be eligible -- per-fold presence is
    handled later in gene_tables(). Also returns, per gene, how many folds
    contain it (for diagnostics)."""
    presence: dict[str, int] = {}
    for fold in range(FOLDS):
        truth = ad.read_h5ad(CASE / SET_NAME / f"fold_{fold}" / "truth.h5ad", backed="r")
        for g in truth.var_names.astype(str):
            presence[g] = presence.get(g, 0) + 1
    return set(presence), presence


def gene_tables(adt_path: Path, keep_genes: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool across folds, RESTRICTED to the mapped genes, joined by SYMBOL (each
    fold has its own gene panel/order -- see module docstring). One row per
    (gene, held-out RNA-zero cell) for the method scores, and one row per
    (gene, RNA-detected cell) for selection."""
    zero_blocks, detected_blocks = [], []
    for fold in range(FOLDS):
        unit = load_unit(SET_NAME, fold, "deploy")
        fold_genes = np.asarray(unit.genes)
        fold_index = {g: i for i, g in enumerate(fold_genes)}
        present = sorted(g for g in keep_genes if g in fold_index)
        if not present:
            print(f"[gene_tables] WARNING: fold {fold} has none of the kept genes -- skipping fold")
            continue
        present_idx = np.asarray([fold_index[g] for g in present], dtype=np.int64)
        cell_ids_all = unit.obs.index.to_numpy()
        donor_all = unit.obs["donor"].to_numpy()

        # --- zero (held-out) candidates, mapped genes only ---
        sf = load_fills(unit, "Safe Fusion")
        rows, cols = sf["rows"], sf["cols"]
        keep_mask = np.isin(cols, present_idx)
        gene_syms = fold_genes[cols[keep_mask]]
        z = {
            "gene": gene_syms,
            "cell_id": cell_ids_all[rows[keep_mask]],
            "donor": donor_all[rows[keep_mask]],
            "Safe Fusion": sf["priority"][keep_mask],
        }
        del sf
        for method in METHODS:
            if method == "Safe Fusion":
                continue
            fills = load_fills(unit, method)
            z[method] = fills["priority"][keep_mask]
            del fills
        zero_blocks.append(pd.DataFrame(z))

        # --- detected (test) cells, mapped genes only ---
        test_rows = np.flatnonzero(unit.test)
        sub = unit.counts[np.ix_(test_rows, present_idx)]      # (n_test x n_present), small
        lib = unit.counts[test_rows].sum(axis=1)
        nz_r, nz_local = np.nonzero(sub > 0)
        present_arr = np.asarray(present)
        detected_blocks.append(pd.DataFrame({
            "gene": present_arr[nz_local],
            "cell_id": cell_ids_all[test_rows][nz_r],
            "donor": donor_all[test_rows][nz_r],
            "rna_cp10k": 1e4 * sub[nz_r, nz_local] / lib[nz_r],
        }))
    zeros = pd.concat(zero_blocks, ignore_index=True)
    detected = pd.concat(detected_blocks, ignore_index=True)
    return zeros, detected


def attach_protein(frame: pd.DataFrame, adt_path: Path, tag_by_gene: dict[str, str]) -> pd.DataFrame:
    """Add the CLR protein for each row's gene-specific antibody tag."""
    needed = sorted({tag for tag in tag_by_gene.values()})
    adt = pd.read_parquet(adt_path, columns=["cell_id"] + [f"clr|{t}" for t in needed]).set_index("cell_id")
    protein = np.full(len(frame), np.nan)
    gn = frame["gene"].to_numpy()
    cids = frame["cell_id"].to_numpy()
    for g, tag in tag_by_gene.items():
        mask = gn == g
        if mask.any():
            protein[mask] = adt.loc[cids[mask], f"clr|{tag}"].to_numpy()
    frame = frame.copy()
    frame["protein"] = protein
    return frame


# --------------------------- stages -----------------------------------------
def stage_select(zeros, detected, tag_map, gene_fold_presence, args, outdir) -> pd.DataFrame:
    kept = tag_map.loc[tag_map["status"] == "kept", ["tag", "gene"]].copy()
    tag_by_gene = dict(zip(kept["gene"], kept["tag"]))

    detected = attach_protein(detected.loc[detected["gene"].isin(tag_by_gene)], args.adt, tag_by_gene)
    detected = detected.loc[np.isfinite(detected["protein"])]
    zeros_counts = zeros.loc[zeros["gene"].isin(tag_by_gene)].groupby("gene").size()

    rows = []
    for gene, tag in tag_by_gene.items():
        d = detected.loc[detected["gene"] == gene]
        n_detected = len(d)
        n_zero = int(zeros_counts.get(gene, 0))
        coexpr = np.nan
        if n_detected >= args.min_detected and d["donor"].nunique() >= 2 and d["protein"].nunique() > 1:
            donors = np.asarray(sorted(d["donor"].unique()))
            w = np.ones((1, len(donors)))
            idx = pd.Index(donors).get_indexer(d["donor"])
            coexpr = float(weighted_spearman(d["rna_cp10k"].to_numpy(float),
                                             d["protein"].to_numpy(float), w[:, idx])[0])
        rows.append({"gene": gene, "tag": tag, "coexpression_spearman": coexpr,
                     "n_detected_cells": n_detected, "n_rna_zero_cells": n_zero,
                     "n_folds_present": gene_fold_presence.get(gene, 0)})
    table = pd.DataFrame(rows).sort_values("coexpression_spearman", ascending=False)

    # the bar: CD274's own coexpression if measured, else --bar
    bar = args.bar
    bar_source = f"--bar={args.bar}"
    if args.bar_gene in set(table["gene"]):
        cd = table.loc[table["gene"] == args.bar_gene, "coexpression_spearman"]
        if np.isfinite(cd.iloc[0]):
            bar = float(cd.iloc[0])
            bar_source = f"{args.bar_gene} coexpression"
    table["bar"] = bar
    table["passes_bar"] = table["coexpression_spearman"] >= bar
    table["enough_zero_cells"] = table["n_rna_zero_cells"] >= args.min_zero_cells
    table["selected"] = table["passes_bar"] & table["enough_zero_cells"] & np.isfinite(table["coexpression_spearman"])

    table.to_csv(outdir / "gene_selection_full.csv", index=False)
    selected = table.loc[table["selected"], ["gene", "tag", "coexpression_spearman",
                                             "n_rna_zero_cells", "n_detected_cells", "n_folds_present"]]
    selected.to_csv(outdir / "selected_genes.csv", index=False)
    meta = {"bar": bar, "bar_source": bar_source, "bar_gene": args.bar_gene,
            "min_zero_cells": args.min_zero_cells, "min_detected": args.min_detected,
            "n_candidate_genes": int(len(table)), "n_selected": int(len(selected)),
            "seed": SEED, "set": SET_NAME}
    (outdir / "selection_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[select] bar ({bar_source}) = {bar:.3f} | candidates {len(table)} -> selected {len(selected)}")
    incomplete = selected.loc[selected["n_folds_present"] < FOLDS]
    if len(incomplete):
        print(f"[select] NOTE: {len(incomplete)} selected gene(s) are present in fewer than "
              f"{FOLDS} folds (less data, still valid): {', '.join(incomplete['gene'])}")
    print(selected.head(40).to_string(index=False))
    return selected


def stage_evaluate(zeros, args, outdir) -> None:
    selected_path = outdir / "selected_genes.csv"
    if not selected_path.exists():
        raise SystemExit("selected_genes.csv missing -- run --stage select first (list must be frozen before scoring).")
    selected = pd.read_csv(selected_path)
    tag_by_gene = dict(zip(selected["gene"], selected["tag"]))

    frame_all = attach_protein(zeros.loc[zeros["gene"].isin(tag_by_gene)], args.adt, tag_by_gene)
    frame_all = frame_all.loc[np.isfinite(frame_all["protein"])]
    methods = list(METHODS)

    per_gene = []
    for gene, tag in tag_by_gene.items():
        f = frame_all.loc[frame_all["gene"] == gene].reset_index(drop=True)
        if len(f) < args.min_zero_cells or f["donor"].nunique() < 2 or f["protein"].nunique() < 2:
            continue
        donors = np.asarray(sorted(f["donor"].unique()))
        donor_index = pd.Index(donors).get_indexer(f["donor"])
        if args.draws > 0:
            weights = np.vstack([np.ones(len(donors)), bootstrap_draws(donors, None, args.draws, SEED)])
        else:
            weights = np.ones((1, len(donors)))
        item_weights = weights[:, donor_index]  # per-cell weights for Spearman
        protein = f["protein"].to_numpy(float)
        rec = {"gene": gene, "tag": tag, "n_rna_zero_cells": int(len(f)),
               "donors": int(len(donors))}
        rho_point, auc_point = {}, {}
        for m in methods:
            score = f[m].to_numpy(float)
            rho = weighted_spearman(score, protein, item_weights)
            auc = threshold_auroc(score, protein, donor_index, donors, weights)
            rho_point[m], auc_point[m] = float(rho[0]), float(auc[0])
            rec[f"spearman|{m}"] = float(rho[0])
            rec[f"auroc|{m}"] = float(auc[0])
            if args.draws > 0:
                rec[f"spearman_ci|{m}"] = interval(rho[1:])
        # Safe Fusion vs each method, paired on the same resample
        if args.draws > 0:
            for m in methods:
                if m == "Safe Fusion":
                    continue
                diff = weighted_spearman(f["Safe Fusion"].to_numpy(float), protein, item_weights) - \
                    weighted_spearman(f[m].to_numpy(float), protein, item_weights)
                rec[f"sf_minus|{m}"] = float(diff[0])
                rec[f"sf_minus_ci|{m}"] = interval(diff[1:])
        win_rho = max(rho_point, key=rho_point.get)
        win_auc = max(auc_point, key=auc_point.get)
        rec["winner_spearman"], rec["winner_auroc"] = win_rho, win_auc
        per_gene.append(rec)

    per_gene = pd.DataFrame(per_gene)
    per_gene.to_csv(outdir / "per_gene_results.csv", index=False)
    n_genes = len(per_gene)
    if n_genes == 0:
        raise SystemExit("No selected gene had enough held-out zero cells / donors / protein variation to score. "
                          "Check selected_genes.csv and --min-zero-cells.")

    # method summary + across-gene sign tests (Safe Fusion vs each)
    summary_rows = []
    for m in methods:
        won_rho = int((per_gene["winner_spearman"] == m).sum())
        won_auc = int((per_gene["winner_auroc"] == m).sum())
        mean_rho = float(per_gene[f"spearman|{m}"].mean())
        row = {"method": m, "genes_won_spearman": won_rho, "genes_won_auroc": won_auc,
               "mean_spearman": mean_rho}
        if m != "Safe Fusion":
            sf_better = (per_gene["spearman|Safe Fusion"] > per_gene[f"spearman|{m}"])
            comparable = (per_gene["spearman|Safe Fusion"] != per_gene[f"spearman|{m}"])
            k, nn = int(sf_better.sum()), int(comparable.sum())
            p = float(stats.binomtest(k, nn, 0.5, alternative="greater").pvalue) if nn else float("nan")
            row["sf_beats_this_genes"] = k
            row["sf_vs_this_tested"] = nn
            row["sf_beats_this_sign_p"] = p
            row["median_sf_minus_this_spearman"] = float(
                (per_gene["spearman|Safe Fusion"] - per_gene[f"spearman|{m}"]).median())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "method_summary.csv", index=False)

    sf_wins_rho = int((per_gene["winner_spearman"] == "Safe Fusion").sum())
    sf_wins_auc = int((per_gene["winner_auroc"] == "Safe Fusion").sum())
    report = {
        "set": SET_NAME, "n_genes_tested": n_genes,
        "safe_fusion_best_spearman_genes": sf_wins_rho,
        "safe_fusion_best_auroc_genes": sf_wins_auc,
        "winners_spearman": {m: int((per_gene["winner_spearman"] == m).sum()) for m in methods},
        "winners_auroc": {m: int((per_gene["winner_auroc"] == m).sum()) for m in methods},
        "draws": args.draws, "seed": SEED,
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        f"Cross-modal multi-gene protein validation (Hao 2021 CITE-seq, '{SET_NAME}' units)",
        f"Genes tested (pre-registered): {n_genes}",
        f"Safe Fusion ranks best (Spearman) for {sf_wins_rho}/{n_genes} genes; "
        f"best (mean threshold AUROC) for {sf_wins_auc}/{n_genes}.",
        "",
        "Genes won per method (Spearman): " +
        ", ".join(f"{m} {int((per_gene['winner_spearman'] == m).sum())}" for m in methods),
        "",
        summary.round(4).to_string(index=False),
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["select", "evaluate", "all"], default="all")
    parser.add_argument("--adt", type=Path, default=CASE / SET_NAME / "adt.parquet")
    parser.add_argument("--output-dir", type=Path, default=CASE / "results" / "q7_protein_multigene")
    parser.add_argument("--bar-gene", default="CD274",
                        help="Gene whose RNA-protein coexpression sets the inclusion bar.")
    parser.add_argument("--bar", type=float, default=0.10,
                        help="Fallback bar if --bar-gene is not in the panel.")
    parser.add_argument("--min-zero-cells", type=int, default=50,
                        help="Minimum held-out RNA-zero cells (with protein) to test a gene.")
    parser.add_argument("--min-detected", type=int, default=100,
                        help="Minimum RNA-detected cells to estimate the selection coexpression.")
    parser.add_argument("--draws", type=int, default=1000,
                        help="Donor bootstrap draws for CIs (0 = point estimates only, fast).")
    args = parser.parse_args()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    gene_set, gene_fold_presence = read_union_genes()
    tag_map = build_tag_gene_map(args.adt, gene_set)
    tag_map.to_csv(outdir / "tag_gene_map.csv", index=False)
    print(f"[map] {int((tag_map['status'] == 'kept').sum())} tags mapped to a unique RNA gene "
          f"out of {len(tag_map)} antibodies.")

    keep_genes = set(tag_map.loc[tag_map["status"] == "kept", "gene"])
    if not keep_genes:
        raise SystemExit("No antibody tag mapped to an RNA gene -- check tag_gene_map.csv and the ADT tag naming.")
    partial = {g for g in keep_genes if gene_fold_presence.get(g, 0) < FOLDS}
    if partial:
        print(f"[map] NOTE: {len(partial)} mapped gene(s) are not present in all {FOLDS} folds "
              f"(different per-fold variable-gene selection): {', '.join(sorted(partial))}")
    zeros, detected = gene_tables(args.adt, keep_genes)

    if args.stage in ("select", "all"):
        stage_select(zeros, detected, tag_map, gene_fold_presence, args, outdir)
    if args.stage in ("evaluate", "all"):
        stage_evaluate(zeros, args, outdir)


if __name__ == "__main__":
    main()
