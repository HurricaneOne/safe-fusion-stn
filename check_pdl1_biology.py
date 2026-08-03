import pandas as pd
import anndata as ad
import numpy as np

# load the real panel and the RNA file (for perturbation labels)
panel = pd.read_csv('data/eccite_validation_panel_real.csv')
a = ad.read_h5ad('data/GSE153056_eccite.h5ad')

# PDL1 protein per cell
pdl1 = panel[panel['protein'] == 'PDL1'].set_index('cell_id')['measurement']

# perturbation label per cell (collapse guide suffix to gene)
import re
pert = a.obs['perturbation'].astype(str)
gene = pert.map(lambda s: 'control' if s == 'control' else re.sub(r'g\d+$', '', s))
gene.index = a.obs_names

# join
df = pd.DataFrame({'pdl1': pdl1})
df['gene'] = gene.reindex(df.index)
df = df.dropna()

ctrl_median = df[df['gene'] == 'control']['pdl1'].median()
print(f"control PDL1 median: {ctrl_median:.1f}\n")

# genes of interest: negative regulators (expect UP), positive regulators (expect DOWN)
watch = ['control', 'CUL3', 'BRD4',            # negative regs -> expect higher
         'IFNGR1', 'IFNGR2', 'IRF1', 'JAK2', 'STAT1']  # positive regs -> expect lower
print(f"{'gene':10s} {'n':>6s} {'PDL1 median':>12s} {'vs control':>12s}")
for g in watch:
    sub = df[df['gene'] == g]['pdl1']
    if len(sub) == 0:
        continue
    med = sub.median()
    direction = 'UP' if med > ctrl_median else 'down'
    print(f"{g:10s} {len(sub):6d} {med:12.1f} {direction:>12s}")