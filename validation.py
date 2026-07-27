"""
validation.py
-------------
The three Perturb-seq validation tests from the mentor's note, each returning a
dict of metrics. Every test scores imputed values ONLY on entries that were zero
in the raw matrix, exactly like the paper's fill-matched CITE-seq protocol.

  Test 1  on_target_precision   -- precision / over-imputation on true KO zeros
  Test 2  downstream_recall     -- preservation of causal downstream DE signal
  Test 3  crossmodal_protein    -- ECCITE-seq surface-protein orthogonal labels

Shared scoring conventions
  score(c,g) = imputed_value(c,g)   (higher => imputer "wants to fill" this zero)
  label 1    = zero that SHOULD be filled (technical dropout)
  label 0    = zero that should be LEFT (true biological absence)
  PR-AUC / ROC-AUC use score as a soft classifier of label; both are rank-based
  so they are invariant to each imputer's output scale.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _log1p_cpm(X):
    X = np.asarray(X, dtype="float64")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, 0.0, None)          # scores can go negative; CPM needs >= 0
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(X / lib * 1e4)


def _fill_matched(scores, labels, tau):
    """Pick threshold so a fraction tau of scored entries are 'filled', then
    report precision/recall/F1 of those fills against the dropout labels."""
    scores = np.nan_to_num(np.asarray(scores, dtype="float64"),
                           nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    labels = np.asarray(labels)
    order = np.argsort(-scores, kind="mergesort")
    n_fill = max(1, int(round(tau * len(scores))))
    filled = np.zeros_like(labels, dtype=bool)
    filled[order[:n_fill]] = True
    tp = int((filled & (labels == 1)).sum())
    fp = int((filled & (labels == 0)).sum())
    fn = int((~filled & (labels == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def _auc_pair(scores, labels):
    scores = np.asarray(scores, dtype="float64")
    labels = np.asarray(labels)
    ok = np.isfinite(scores)
    scores, labels = scores[ok], labels[ok]
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


# --------------------------------------------------------------------------- #
# Test 1 -- On-target knockdown: true biological zeros vs dropout zeros
# --------------------------------------------------------------------------- #
def test1_on_target(adata, imputed, tau=0.081, min_ctrl_expr_frac=0.05):
    """
    True zeros  (label 0): targeted-TF gene is zero in cells carrying its guide.
                           Knockdown makes this a genuine biological zero.
    Dropout     (label 1): same TF gene is zero in NON-TARGETING control cells,
                           for TFs that are expressed across the control pool
                           (so a zero there is a technical dropout candidate).
    A precise imputer keeps score low on label-0 (does NOT fill knockouts) and
    higher on label-1. We report ROC/PR-AUC plus the over-imputation rate: the
    fraction of certified knockout zeros that get filled at fill rate tau.
    """
    X = np.asarray(adata.X, dtype="float32")
    imp = np.asarray(imputed, dtype="float32")
    guide = np.asarray(adata.obs["guide_target"])
    var = {g: j for j, g in enumerate(adata.var_names)}
    tfs = sorted(set(guide) - {"NT"})
    ctrl = guide == "NT"

    scores, labels, ko_scores = [], [], []
    for tf in tfs:
        if tf not in var:
            continue
        j = var[tf]
        # only use TFs that are genuinely expressed in controls
        if (X[ctrl, j] > 0).mean() < min_ctrl_expr_frac:
            continue
        ko = (guide == tf) & (X[:, j] == 0)          # certified true zeros
        dp = ctrl & (X[:, j] == 0)                    # dropout candidates
        for c in np.where(ko)[0]:
            scores.append(imp[c, j]); labels.append(0); ko_scores.append(imp[c, j])
        for c in np.where(dp)[0]:
            scores.append(imp[c, j]); labels.append(1)

    scores = np.asarray(scores, dtype="float64")
    labels = np.asarray(labels, dtype="int64")
    roc, pr = _auc_pair(scores, labels)
    prec, rec, f1 = _fill_matched(scores, labels, tau)

    # over-imputation (the SafeFusion-relevant number): fraction of certified
    # knockout zeros that the imputer assigned ANY positive value. This is the
    # false-positive fill rate on true biological zeros. Lower = more precise;
    # a method that never fills knockouts scores 0.0.
    ko = np.asarray(ko_scores, dtype="float64")
    ko_fill = float((ko > 1e-6).mean()) if ko.size else float("nan")

    return {"test": "1_on_target", "n_pos": int((labels == 1).sum()),
            "n_neg": int((labels == 0).sum()), "roc_auc": roc, "pr_auc": pr,
            "prec@tau": prec, "rec@tau": rec, "f1@tau": f1,
            "ko_fill_rate": ko_fill}


# --------------------------------------------------------------------------- #
# Test 2 -- Downstream causal effects: is the knockdown DE signal preserved?
# --------------------------------------------------------------------------- #
def test2_downstream(adata, imputed):
    """
    For each perturbed TF with a known target module, knockdown should SUPPRESS
    those targets (negative logFC vs control). Over-smoothing blends KO cells
    back to the population mean and erases that signal.

    Metrics:
      lfc_corr        Pearson r between raw logFC and imputed logFC over targets
                      (1.0 = signal shape fully preserved).
      signal_retained mean(imputed logFC) / mean(raw logFC) over targets
                      (1.0 = magnitude preserved, ~0 = erased, <0 = flipped).
      target_roc/pr   rank all genes by |imputed KO-vs-control logFC|; label 1 if
                      the gene is a true target of the perturbed TF. AUC = how
                      well the post-imputation data still recovers the module.
    """
    guide = np.asarray(adata.obs["guide_target"])
    tf_targets = adata.uns.get("tf_targets", {}) if hasattr(adata.uns, "get") else dict(adata.uns).get("tf_targets", {})
    if not tf_targets:
        return {"test": "2_downstream", "n_tfs": 0, "lfc_corr": float("nan"),
                "signal_retained": float("nan"), "target_roc_auc": float("nan"),
                "target_pr_auc": float("nan"),
                "note": "no tf_targets in adata.uns; supply curated downstream modules"}
    var = {g: j for j, g in enumerate(adata.var_names)}
    ctrl = guide == "NT"

    Lraw = _log1p_cpm(adata.X)
    Limp = _log1p_cpm(imputed)
    ctrl_raw = Lraw[ctrl].mean(0)
    ctrl_imp = Limp[ctrl].mean(0)

    raw_lfc_all, imp_lfc_all, is_target_all = [], [], []
    per_tf = {}
    for tf, targets in tf_targets.items():
        cells = guide == tf
        if cells.sum() < 5:
            continue
        raw_lfc = Lraw[cells].mean(0) - ctrl_raw
        imp_lfc = Limp[cells].mean(0) - ctrl_imp
        tcols = [var[t] for t in targets if t in var]
        if not tcols:
            continue
        r_t = raw_lfc[tcols]; i_t = imp_lfc[tcols]
        per_tf[tf] = {
            "lfc_corr": float(np.corrcoef(r_t, i_t)[0, 1]) if len(tcols) > 1 else float("nan"),
            "signal_retained": float(i_t.mean() / r_t.mean()) if r_t.mean() != 0 else float("nan"),
        }
        raw_lfc_all.append(raw_lfc); imp_lfc_all.append(imp_lfc)
        lab = np.zeros(adata.n_vars); lab[tcols] = 1
        is_target_all.append(lab)

    corr = float(np.nanmean([v["lfc_corr"] for v in per_tf.values()])) if per_tf else float("nan")
    retained = float(np.nanmean([v["signal_retained"] for v in per_tf.values()])) if per_tf else float("nan")

    if imp_lfc_all:
        scores = np.abs(np.concatenate(imp_lfc_all))
        labels = np.concatenate(is_target_all).astype(int)
        roc, pr = _auc_pair(scores, labels)
    else:
        roc = pr = float("nan")

    return {"test": "2_downstream", "n_tfs": len(per_tf), "lfc_corr": corr,
            "signal_retained": retained, "target_roc_auc": roc, "target_pr_auc": pr}


# --------------------------------------------------------------------------- #
# Test 3 -- Cross-modal ECCITE-seq surface-protein validation (paper Eq. 6/7)
# --------------------------------------------------------------------------- #
def test3_crossmodal(adata, imputed, tau=0.081, theta_quantile=0.5, theta_abs=None):
    """
    For each aligned (gene g, surface protein p): among cells where RNA g is zero,
        label 1 (dropout)      if protein p is detectable (> threshold)
        label 0 (true absence) if protein p is below threshold
    Score = imputed RNA value. Reports PR-AUC, ROC-AUC and fill-matched
    precision/recall/F1 at tau -- identical in spirit to the paper's CITE-seq
    validation, now on ECCITE-seq.

    theta_abs: if set (e.g. 0 for a binary 0/1 panel), a protein counts as
    detectable when value > theta_abs. Otherwise use the theta_quantile quantile.
    """
    X = np.asarray(adata.X, dtype="float32")
    imp = np.asarray(imputed, dtype="float32")
    prot = np.asarray(adata.obsm["protein"], dtype="float32")
    pnames = list(adata.uns["protein_names"])
    align = dict(adata.uns["rna_for_protein"])
    var = {g: j for j, g in enumerate(adata.var_names)}

    scores, labels = [], []
    for pi, pname in enumerate(pnames):
        gene = align[pname]
        if gene not in var:
            continue
        j = var[gene]
        theta = theta_abs if theta_abs is not None else np.quantile(prot[:, pi], theta_quantile)
        zero = X[:, j] == 0
        for c in np.where(zero)[0]:
            pv = prot[c, pi]
            if np.isnan(pv):
                continue
            scores.append(imp[c, j])
            labels.append(1 if pv > theta else 0)

    scores = np.asarray(scores, dtype="float64")
    labels = np.asarray(labels, dtype="int64")
    roc, pr = _auc_pair(scores, labels)
    prec, rec, f1 = _fill_matched(scores, labels, tau)
    return {"test": "3_crossmodal", "n_pos": int((labels == 1).sum()),
            "n_neg": int((labels == 0).sum()), "roc_auc": roc, "pr_auc": pr,
            "prec@tau": prec, "rec@tau": rec, "f1@tau": f1}


def run_all(adata, imputed, tau=0.081, theta_abs=None, min_ctrl_expr_frac=0.05):
    return [test1_on_target(adata, imputed, tau=tau, min_ctrl_expr_frac=min_ctrl_expr_frac),
            test2_downstream(adata, imputed),
            test3_crossmodal(adata, imputed, tau=tau, theta_abs=theta_abs)]
