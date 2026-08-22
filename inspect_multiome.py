import scanpy as sc
import numpy as np

# the 10x .h5 holds BOTH modalities; gex_only=False loads all feature types
adata = sc.read_10x_h5('data/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5',
                       gex_only=False)
adata.var_names_make_unique()

print("full object:", adata.shape)
print("\nfeature types present:")
print(adata.var['feature_types'].value_counts())

# split by modality
is_rna = adata.var['feature_types'] == 'Gene Expression'
is_atac = adata.var['feature_types'] == 'Peaks'
print(f"\nRNA genes: {is_rna.sum()}")
print(f"ATAC peaks: {is_atac.sum()}")
print(f"cells: {adata.n_obs}")

# what does a peak name look like? (this tells us the coordinate format)
print("\nexample RNA gene names:", list(adata.var_names[is_rna][:5]))
print("example ATAC peak names:", list(adata.var_names[is_atac][:5]))

# is there gene annotation with genomic coordinates? (needed for peak->gene)
print("\nvar columns:", list(adata.var.columns))
print(adata.var[is_rna].head(3))