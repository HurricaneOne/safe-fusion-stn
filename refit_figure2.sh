#!/usr/bin/env bash
# =============================================================================
# refit_figure2.sh  —  Refit the Figure 2 (Papalexi CD274 / PD-L1 cross-modal)
# models at a chosen seed, HOLDING THE TEST SET FIXED (Option A), then evaluate,
# plot, and diff against the shipped manuscript numbers.
#
# Run from the repository root (safe-fusion-stn), on WSL2 with your RTX 4070.
# The masked benchmark (corrupted.h5ad + splits) is reused unchanged from the
# released LFS tarball, so the held-out cells are identical to the manuscript
# and the differences reflect model-refitting at the new seed, not a new split.
#
# USAGE
#   ./refit_figure2.sh <SEED> [OUTDIR]
#
# RECOMMENDED: run twice — a GPU control at the original seed, then the new seed:
#   ./refit_figure2.sh 1729 refit_control_1729
#   ./refit_figure2.sh 2718 refit_seed_2718
#   python diff_vs_reference.py refit_control_1729    # recipe/GPU gap at 1729
#   python diff_vs_reference.py refit_seed_2718       # the new-seed result
#
# ENVIRONMENT (single env is fine; see README_WSL2.md to build it):
#   PY        python with torch(CUDA)+scvi-tools==1.2.2.post2+magic-impute+pkg
#             (default: .venv/bin/python)
#   SCVI_PY   python for the scVI step   (default: $PY)
#   MAGIC_PY  python for the MAGIC step  (default: $PY)
# =============================================================================
set -euo pipefail

SEED="${1:?usage: ./refit_figure2.sh <SEED> [OUTDIR]}"
OUT="${2:-refit_seed_${SEED}}"

PY="${PY:-.venv/bin/python}"
SCVI_PY="${SCVI_PY:-$PY}"
MAGIC_PY="${MAGIC_PY:-$PY}"

SRC=artifacts/paper_evidence/papalexi_crossmodal/benchmark
PREPARED=external_data/prepared/papalexi_eccite_crossmodal.h5ad
PANEL=artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet

export TQDM_DISABLE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
       OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
       MPLBACKEND=Agg MPLCONFIGDIR="/tmp/mpl_${SEED}"

mkdir -p "$OUT"
PROV="$OUT/provenance.txt"
common=(--input "$SRC/corrupted.h5ad" --coordinates "$SRC/coordinates.parquet" --splits "$SRC/splits.parquet" --seed "$SEED")
teachers=(--teacher-contract "$OUT/gene_median" --teacher-contract "$OUT/svd_impute" \
          --teacher-contract "$OUT/graph_smooth" --teacher-contract "$OUT/magic" \
          --teacher-contract "$OUT/scvi_counts")

# --- sanity: inputs present ---
for f in "$SRC/corrupted.h5ad" "$SRC/splits.parquet" "$SRC/coordinates.parquet" "$PREPARED" "$PANEL"; do
  [[ -e "$f" ]] || { echo "ERROR: missing input $f (did you extract the tarball at repo root?)"; exit 1; }
done

step() {  # step "name" cmd...
  local name="$1"; shift
  local t0 t1
  echo "[$(date +%H:%M:%S)] START $name"
  t0=$(date +%s)
  "$@" > "$OUT/log_${name}.txt" 2>&1
  t1=$(date +%s)
  printf '%-22s %6ss\n' "$name" "$((t1 - t0))" >> "$PROV"
  echo "[$(date +%H:%M:%S)] done  $name  ($((t1 - t0))s)"
}

# ---------------------------------------------------------------- provenance
{
  echo "=== refit_figure2.sh provenance ==="
  echo "date              : $(date -u +%FT%TZ)"
  echo "seed (refit)      : $SEED"
  echo "eval seed (fixed) : 1729"
  echo "outdir            : $OUT"
  echo "git HEAD          : $(git rev-parse HEAD 2>/dev/null || echo n/a)"
  echo "git branch        : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
  echo "python            : $($PY -c 'import sys;print(sys.version.split()[0])')"
  echo "torch             : $($PY -c 'import torch;print(torch.__version__)' 2>/dev/null || echo n/a)"
  echo "cuda available    : $($PY -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null || echo n/a)"
  echo "gpu               : $($PY -c 'import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")' 2>/dev/null || echo n/a)"
  echo "scvi-tools        : $($SCVI_PY -c 'import scvi;print(scvi.__version__)' 2>/dev/null || echo n/a)"
  echo "nvidia-smi        :"
  (nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  n/a") | sed 's/^/  /'
  echo
  echo "=== per-step wall time ==="
} > "$PROV"

# ------------------------------------------------------------------- pipeline
# 1. scVI teacher (GPU)  — original script, accelerator=\"gpu\"
step scvi "$SCVI_PY" scripts/run_scvi_baseline.py \
  --corrupted "$SRC/corrupted.h5ad" --coordinates "$SRC/coordinates.parquet" \
  --splits "$SRC/splits.parquet" --output "$OUT/scvi" --epochs 200 --seed "$SEED"

# 2. CPU teachers (deterministic given splits; reseeded for completeness)
for m in gene_median svd_impute graph_smooth; do
  step "teacher_$m" "$PY" scripts/run_leakage_safe_method.py --method "$m" --transductive "${common[@]}" --output "$OUT/$m"
done

# 3. MAGIC, then count-scale MAGIC and scVI onto the corrupted count scale
step magic "$MAGIC_PY" scripts/run_magic_baseline.py \
  --corrupted "$SRC/corrupted.h5ad" --coordinates "$SRC/coordinates.parquet" \
  --splits "$SRC/splits.parquet" --output "$OUT/magic_standard" --n-jobs "${OMP_NUM_THREADS:-8}" --seed "$SEED"
step magic_cs "$PY" scripts/count_scale_contract.py --contract "$OUT/magic_standard" --corrupted "$SRC/corrupted.h5ad" --output "$OUT/magic"
step scvi_cs  "$PY" scripts/count_scale_contract.py --contract "$OUT/scvi"          --corrupted "$SRC/corrupted.h5ad" --output "$OUT/scvi_counts"

# 4. Fusion (gradient-boosted safe_fusion over the five teachers)
step fusion "$PY" scripts/run_leakage_safe_method.py --method safe_fusion --transductive "${common[@]}" \
  --output "$OUT/safe_fusion" "${teachers[@]}"

# 5. Selector MLP
step selector "$PY" scripts/calibrated_selective_fill.py \
  --corrupted "$SRC/corrupted.h5ad" --truth "$PREPARED" \
  --coordinates "$SRC/coordinates.parquet" --splits "$SRC/splits.parquet" \
  --fusion-contract "$OUT/safe_fusion" "${teachers[@]}" \
  --output-dir "$OUT/mlp_selector" --fit-split development --architecture mlp \
  --budget-mode apply_topk --budgets 0.02 \
  --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 \
  --score-gene CD274 CD86 PDCD1LG2 HAVCR2 --seed "$SEED"

# 6. Evaluate CD274 on CLR and raw counts (eval seed fixed at 1729: identical CI method)
for column in adt_clr adt_count; do
  outdir="$OUT/evaluation_cd274"; [[ "$column" == adt_count ]] && outdir="$OUT/evaluation_cd274_raw_counts"
  step "eval_$column" "$PY" scripts/evaluate_papalexi_crossmodal.py \
    --prepared "$PREPARED" --corrupted "$SRC/corrupted.h5ad" --panel "$PANEL" \
    --selector-scores "$OUT/mlp_selector/selected_gene_scores.parquet" \
    --fusion-mean "$OUT/safe_fusion/mean.npy" --graph-mean "$OUT/graph_smooth/mean.npy" \
    --svd-mean "$OUT/svd_impute/mean.npy" --scvi-mean "$OUT/scvi/mean.npy" \
    --output-dir "$outdir" --protein-column "$column" --gene CD274 \
    --bootstrap 1000 --permutations 1000 --seed 1729
done

# 6b. PD-L1 within-state baselines (best-effort; not required for the Figure 2 summary)
step pdl1_state "$PY" scripts/evaluate_pdl1_state_baselines.py \
  --prepared "$PREPARED" --benchmark-root "$OUT" --panel "$PANEL" \
  --output-dir "$OUT/evaluation_pdl1_state" --bootstrap 1000 --seed 1729 || \
  echo "(pdl1_state step skipped/failed — not needed for the Figure 2 method summary)"

# 7. Plot Figure 2
step plot "$PY" scripts/plot_biological_range_figures.py --pdl1-only \
  --crossmodal-root "$OUT/evaluation_cd274" --output-dir "$OUT/figures" --paper-dir "$OUT/figures"

# 8. Diff vs the shipped manuscript numbers
echo
echo "================= CD274 method summary: refit vs manuscript ================="
"$PY" diff_vs_reference.py "$OUT"

echo
echo "Done. Key outputs under $OUT/:"
echo "  evaluation_cd274/method_summary.csv   (your refit numbers)"
echo "  diff_cd274.csv                        (refit vs manuscript, with diffs)"
echo "  figures/pdl1_range_validation.png     (Figure 2)"
echo "  provenance.txt                        (hardware + per-step runtime)"
