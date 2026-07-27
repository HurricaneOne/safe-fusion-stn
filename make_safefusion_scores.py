#!/usr/bin/env python3
"""
make_safefusion_scores.py
-------------------------
Assemble the gated safe-fusion score s_safe (paper Eq. 1-4) into a cells x genes
matrix that plugs into the Perturb-seq harness.

PREFERRED: if the repo can already dump its own final safe-fusion score, use that
directly (it preserves the repo's exact normalization). Use THIS script only when
the repo exposes the ingredients (posterior mean mu, posterior variance v, and a
fallback-teacher score) but not the final gated matrix.

Inputs (all cells x genes, aligned to the SAME raw matrix X):
  mu        posterior mean in COUNT space              (from the latent-truth model)
  v         posterior variance in LOG space            (from the latent-truth model)
  s_best    fallback score = max over fallback teachers (Eq. 3); usually scVI's
            log1p-CPM. If you only have scVI, pass scVI's log1p-CPM as s_best.
  X         raw counts (defines the zero set Z)
Params:
  q         confidence quantile for the gate (paper default 0.90)
  eps       small constant in Eq. 1 (paper default 1.0)
  v_clip    clip variance to avoid -log(0) blowups
"""
import numpy as np, argparse


def safe_fusion_score(mu, v, s_best, X, q=0.90, eps=1.0, v_clip=(1e-6, 1e6),
                      score_mode="expression"):
    mu = np.asarray(mu, float); v = np.asarray(v, float)
    s_best = np.asarray(s_best, float); X = np.asarray(X, float)
    assert mu.shape == v.shape == s_best.shape == X.shape, \
        f"shape mismatch: mu{mu.shape} v{v.shape} s_best{s_best.shape} X{X.shape}"

    v = np.clip(v, *v_clip)
    conf = -np.log(v)                                      # Eq. 2
    Z = (X == 0)                                           # zero set
    eta_q = np.quantile(conf[Z], q)                        # gate threshold on Z
    use_fuse = conf >= eta_q
    coverage = float((use_fuse & Z).sum() / Z.sum())       # Eq. 5

    if score_mode == "zscore":
        # variance-normalized latent z-score (ranking score), Eq. 1 + gate Eq. 4
        s_fuse = (np.log1p(mu) - np.log1p(eps)) / np.sqrt(v)
        out = np.where(use_fuse, s_fuse, s_best)
    elif score_mode == "expression":
        # the EXPRESSION matrix safe-fusion produces: keep observed counts, and
        # on zero entries fill with posterior mean where the gate is confident,
        # otherwise fall back to the teacher (both ~count scale). Coherent for
        # fold-change based tests (Test 2) and faithful for Tests 1/3.
        out = X.copy()
        out[Z & use_fuse] = mu[Z & use_fuse]
        out[Z & ~use_fuse] = s_best[Z & ~use_fuse]
    else:
        raise ValueError(f"unknown score_mode: {score_mode}")

    return out, {"eta_q": float(eta_q), "coverage": coverage,
                 "n_zero": int(Z.sum()), "mode": score_mode}


def _load(p):
    if p.endswith(".h5ad"):
        import anndata as ad
        from scipy import sparse
        X = ad.read_h5ad(p).X
        return np.asarray(X.todense() if sparse.issparse(X) else X, dtype="float64")
    return np.load(p) if p.endswith(".npy") else np.loadtxt(p, delimiter=",")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", required=True); ap.add_argument("--v", required=True)
    ap.add_argument("--s_best", required=True); ap.add_argument("--X", required=True)
    ap.add_argument("--q", type=float, default=0.90); ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--score_mode", choices=["expression", "zscore"], default="expression",
                    help="expression = the matrix safe-fusion produces (use for all tests); "
                    "zscore = ranking score")
    ap.add_argument("--out", default="safefusion_scores.npy")
    a = ap.parse_args()
    s_safe, info = safe_fusion_score(_load(a.mu), _load(a.v), _load(a.s_best), _load(a.X),
                                     q=a.q, eps=a.eps, score_mode=a.score_mode)
    np.save(a.out, s_safe)
    print(f"wrote {a.out}  shape={s_safe.shape}  {info}")
