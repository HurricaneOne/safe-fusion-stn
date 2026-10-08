@echo off
set "ROOT=artifacts\paper_evidence\papalexi_crossmodal_full\benchmark"
set "PREP=external_data\prepared\papalexi_eccite_crossmodal_full.h5ad"
set "INPUT_RAW=external_data\scperturb\PapalexiSatija2021_eccite_RNA.h5ad"

if not exist "%ROOT%" mkdir "%ROOT%"

echo ===================================================================
echo [1/6] Preparing full dataset...
echo ===================================================================
python scripts\prepare_external_perturbseq.py --input "%INPUT_RAW%" --output "%PREP%" --dataset papalexi_eccite --include-gene CD274 CD86 PDCD1LG2 HAVCR2 --max-cells-per-condition 1000000 --max-control-cells 1000000 --max-genes 2048 --seed 1729
if errorlevel 1 exit /b %errorlevel%

echo ===================================================================
echo [2/6] Building masked benchmark...
echo ===================================================================
python scripts\build_papalexi_crossmodal_benchmark.py --input "%PREP%" --output-dir "%ROOT%" --seed 1729
if errorlevel 1 exit /b %errorlevel%

echo ===================================================================
echo [3/6] Running teacher imputers...
echo ===================================================================
for %%m in (gene_median svd_impute graph_smooth) do (
    echo Running teacher: %%m ...
    python scripts\run_leakage_safe_method.py --method %%m --transductive --input "%PREP%" --coordinates "%ROOT%\coordinates.parquet" --splits "%ROOT%\splits.parquet" --output "%ROOT%\%%m" --seed 1729
)

echo ===================================================================
echo [4/6] Running MAGIC baseline...
echo ===================================================================
python scripts\run_magic_baseline.py --corrupted "%ROOT%\corrupted.h5ad" --coordinates "%ROOT%\coordinates.parquet" --splits "%ROOT%\splits.parquet" --output "%ROOT%\magic_inductive" --seed 1729

echo ===================================================================
echo [5/6] Running scVI baseline...
echo ===================================================================
python scripts\run_scvi_baseline.py --corrupted "%ROOT%\corrupted.h5ad" --coordinates "%ROOT%\coordinates.parquet" --splits "%ROOT%\splits.parquet" --output "%ROOT%\scvi" --epochs 200 --seed 1729

echo ===================================================================
echo [6/6] Running SafeFusion Boosted and MLP Selector...
echo ===================================================================
python scripts\run_leakage_safe_method.py --method safe_fusion --input "%PREP%" --coordinates "%ROOT%\coordinates.parquet" --splits "%ROOT%\splits.parquet" --output "%ROOT%\safe_fusion" --value-model boosted --seed 1729 --teacher "%ROOT%\gene_median" --teacher "%ROOT%\svd_impute" --teacher "%ROOT%\graph_smooth"

python scripts\calibrated_selective_fill.py --corrupted "%ROOT%\corrupted.h5ad" --truth "%PREP%" --coordinates "%ROOT%\coordinates.parquet" --splits "%ROOT%\splits.parquet" --fusion-contract "%ROOT%\safe_fusion" --teacher "%ROOT%\gene_median" --teacher "%ROOT%\svd_impute" --teacher "%ROOT%\graph_smooth" --output-dir "%ROOT%\mlp_selector" --fit-split development --architecture mlp --budget-mode apply_topk --budgets 0.02 --score-gene CD274 CD86 PDCD1LG2 HAVCR2 --seed 1729

echo Pipeline finished successfully!