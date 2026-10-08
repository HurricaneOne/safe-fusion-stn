#!/usr/bin/env python3
"""Check the 7 Hao 2021 GEO download files for gzip corruption/truncation.
Run from the repo root: python check_hao_files.py
"""
import gzip
import os

FILES = [
    "GSM5008737_RNA_3P-matrix.mtx.gz",
    "GSM5008737_RNA_3P-barcodes.tsv.gz",
    "GSM5008737_RNA_3P-features.tsv.gz",
    "GSM5008738_ADT_3P-matrix.mtx.gz",
    "GSM5008738_ADT_3P-barcodes.tsv.gz",
    "GSM5008738_ADT_3P-features.tsv.gz",
    "GSE164378_sc.meta.data_3P.csv.gz",
]

bad = []
for name in FILES:
    path = os.path.join("external_data", "citeseq_hao2021", name)
    if not os.path.exists(path):
        print("MISSING", name)
        bad.append(name)
        continue
    size = os.path.getsize(path)
    try:
        with gzip.open(path, "rb") as fh:
            n = 0
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                n += len(chunk)
        print(f"OK    {name}  ({size} bytes on disk, {n} bytes decompressed)")
    except Exception as e:
        print(f"BAD   {name}  ({size} bytes on disk)  {type(e).__name__}: {e}")
        bad.append(name)

print()
if bad:
    print("Re-download these:", ", ".join(bad))
else:
    print("All 7 files OK.")
