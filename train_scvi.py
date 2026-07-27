import scanpy as sc
import scvi
from pathlib import Path
import numpy as np

print("Loading filtered dataset...")
adata = sc.read_h5ad("data/eccite_filtered.h5ad")

print("Setting up scVI model (preserving full matrix shape)...")
scvi.model.SCVI.setup_anndata(adata)

print("Training deep generative model...")
model = scvi.model.SCVI(adata, n_latent=10, n_layers=2)
model.train(max_epochs=100, batch_size=128)

print("Extracting continuous imputed matrix...")
imputed_matrix = model.get_normalized_expression(library_size=10000)
log_imputed = np.log1p(imputed_matrix).astype(np.float32)

out_path = Path("output/scVI/eccite_filtered/control_vs_knockdown")
out_path.mkdir(parents=True, exist_ok=True)
np.save(out_path / "pbmc.npy", log_imputed)
print("SUCCESS: Full-sized continuous matrix saved!")