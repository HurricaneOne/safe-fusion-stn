import mudata as md, anndata as ad, re
import collections

m = md.read_h5mu('data/papalexi_2021.h5mu')
a = ad.read_h5ad('data/GSE153056_eccite.h5ad')
adt = m.mod['adt']

# strip lane prefix (l1_, l2_, ...) from mudata barcodes
stripped = [re.sub(r'^l\d+_', '', b) for b in adt.obs_names]
c = collections.Counter(stripped)

# keep only barcodes that appear in exactly one lane (no collisions)
keep = {k for k, v in c.items() if v == 1}
m_map = {re.sub(r'^l\d+_', '', b): i
         for i, b in enumerate(adt.obs_names)
         if re.sub(r'^l\d+_', '', b) in keep}

# guide label column may be 'guide_ID' or 'guide_id'
if 'guide_ID' in adt.obs:
    m_guide = adt.obs['guide_ID'].values
else:
    m_guide = adt.obs['guide_id'].values
a_guide = a.obs['guide_id'].astype(str).values

agree = tot = 0
for ai, bc in enumerate(a.obs_names):
    if bc in m_map:
        tot += 1
        if str(m_guide[m_map[bc]]) == str(a_guide[ai]):
            agree += 1

print(f'matched cells: {tot}')
print(f'guide-label agreement: {agree}/{tot} = {agree/tot:.4f}' if tot else 'no matches')