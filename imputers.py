"""
imputers.py
-----------
Every imputer here follows ONE contract:

    impute(X_raw: np.ndarray[cells, genes]) -> np.ndarray[cells, genes]

The returned matrix is the imputed expression the validation tests will score.
The validation only ever looks at entries that were ZERO in X_raw, so an imputer
that leaves nonzeros untouched is fine.

WIRE IN YOUR REAL METHODS HERE:
  - safefusion_impute: replace the body with a call into safefusion-stn so it
    returns the safe-fusion expression score (Eq. 1/4 in the paper) per entry.
  - scvi_impute: uses scvi-tools if installed (the paper uses scVI as a teacher
    and conservative fallback, so it MUST appear as a baseline in every test).

The three reference imputers (identity / oversmooth / conservative) need no
external deps and exist so the harness can be verified without the real models.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------------------------------- #
# Plug-points for the real methods
# --------------------------------------------------------------------------- #
def safefusion_impute(X_raw, score_path="safefusion_scores.npy", **kwargs):
    """
    Loads the precomputed safe-fusion score matrix s_safe (cells x genes, aligned
    to X_raw). Produce it one of two ways:
      (A) dump the repo's own final safe-fusion score to safefusion_scores.npy, OR
      (B) run make_safefusion_scores.py on the repo's mu / v / fallback outputs.
    Only originally-zero entries are scored downstream, so nonzero entries are
    irrelevant here.
    """
    import numpy as np
    s = np.load(score_path)
    X = np.asarray(X_raw, dtype="float32")
    assert s.shape == X.shape, (
        f"score matrix {s.shape} does not match X {X.shape}. If the repo is "
        f"genes x cells, transpose before saving.")
    return s.astype("float32")


def scvi_impute(X_raw, n_latent=10, max_epochs=100, **kwargs):
    """
    scVI baseline (required in every test as a reference / teacher).
    Uses scvi-tools if available; returns the decoded expected expression.
    """
    import anndata as ad
    import scvi
    a = ad.AnnData(X=np.asarray(X_raw).astype("float32"))
    scvi.model.SCVI.setup_anndata(a)
    model = scvi.model.SCVI(a, n_latent=n_latent)
    model.train(max_epochs=max_epochs)
    # Expected normalized expression, scaled back to library size ~ a proxy fill.
    norm = model.get_normalized_expression(a, library_size="latent")
    return np.asarray(norm, dtype="float32")


# --------------------------------------------------------------------------- #
# Reference imputers (no external deps) -- for verifying the harness
# --------------------------------------------------------------------------- #
def identity_impute(X_raw, **kwargs):
    """No imputation. Lower bound: never fills a zero."""
    return np.asarray(X_raw, dtype="float32").copy()


def _knn_graph(Xln, k):
    nn = NearestNeighbors(n_neighbors=min(k + 1, Xln.shape[0])).fit(Xln)
    _, idx = nn.kneighbors(Xln)
    return idx[:, 1:]  # drop self


def oversmooth_impute(X_raw, k=30, **kwargs):
    """
    Aggressive KNN averaging: fills essentially every zero with the neighborhood
    mean. Stands in for over-smoothers (MAGIC / naive autoencoders). Should score
    POORLY on the on-target test and ERASE downstream signal -- by design.
    """
    X = np.asarray(X_raw, dtype="float32")
    Xln = np.log1p(X)
    idx = _knn_graph(Xln, k)
    out = X.copy()
    neigh_mean = np.expm1(Xln[idx].mean(axis=1))
    zero = X == 0
    out[zero] = neigh_mean[zero]
    return out


def conservative_impute(X_raw, k=30, min_frac=0.6, **kwargs):
    """
    A crude SafeFusion-like analog: only fills a zero when a strong majority of
    neighbors express the gene (a confidence gate), otherwise leaves it at zero.
    Should PRESERVE knockout zeros better and keep downstream signal.
    """
    X = np.asarray(X_raw, dtype="float32")
    Xln = np.log1p(X)
    idx = _knn_graph(Xln, k)
    out = X.copy()
    neigh = X[idx]                      # cells x k x genes
    frac_expr = (neigh > 0).mean(axis=1)      # fraction of neighbors expressing
    neigh_mean = neigh.mean(axis=1)
    confident = frac_expr >= min_frac
    zero = X == 0
    fill = zero & confident
    out[fill] = neigh_mean[fill]
    return out


REGISTRY = {
    "safefusion": safefusion_impute,
    "scvi": scvi_impute,
    "identity": identity_impute,
    "oversmooth": oversmooth_impute,
    "conservative": conservative_impute,
}
