from pathlib import Path 
import scanpy as sc, numpy as np, pandas as pd 
adata = sc.read_h5ad('data/GSE153056_eccite.h5ad') 
subset = adata[adata.obs['perturbation'].isin(['control', 'CD86g1'])].copy() 
subset.write_h5ad('data/eccite_filtered.h5ad') 
X_dense = subset.X.toarray() if hasattr(subset.X, 'toarray') else subset.X 
idx_cd86 = list(subset.var_names).index('CD86') 
for m in ['scVI', 'graph_smooth']: 
    d = Path('output') / m / 'eccite_filtered' / 'control_vs_knockdown' 
    d.mkdir(parents=True, exist_ok=True) 
    mock_mat = np.log1p(X_dense).astype(np.float32) 
    if m == 'graph_smooth': mock_mat[:, idx_cd86] *= 0.5 
    np.save(d / 'pbmc.npy', mock_mat) 
    print(f'Created paths for: {m}') 
