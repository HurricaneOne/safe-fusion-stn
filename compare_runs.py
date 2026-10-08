import pandas as pd
base = r"artifacts\paper_evidence\papalexi_crossmodal\benchmark"
pairs = [
    ("CLR",        r"\evaluation_cd274",            r"\evaluation_cd274_6a95bbd"),
    ("raw_counts", r"\evaluation_cd274_raw_counts", r"\evaluation_cd274_raw_counts_6a95bbd"),
]
for label, ref_dir, mine_dir in pairs:
    try:
        ref  = pd.read_csv(base + ref_dir  + r"\method_summary.csv").set_index("method")
        mine = pd.read_csv(base + mine_dir + r"\method_summary.csv").set_index("method")
    except FileNotFoundError as e:
        print(f"[{label}] missing file: {e}\n")
        continue
    print(f"=== {label} ===")
    cols = [c for c in ref.select_dtypes("number").columns if c != "n_rna_zero_cells"]
    rows = [{"method": m, "metric": c,
             "mine": mine.loc[m, c], "his": ref.loc[m, c],
             "diff": mine.loc[m, c] - ref.loc[m, c]}
            for m in ref.index for c in cols]
    print(pd.DataFrame(rows).to_string(index=False))
    print()