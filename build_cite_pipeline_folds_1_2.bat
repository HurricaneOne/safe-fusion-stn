@echo off
REM ============================================================
REM Runs the cite-fold pipeline for folds 1 and 2 ONLY. Use this
REM when fold 0 already completed (or was finished manually) and
REM you don't want to redo its scVI/MAGIC compute.
REM Same stages, same safety checks as build_cite_pipeline.bat.
REM ============================================================

setlocal enabledelayedexpansion

set ROOT=artifacts\paper_evidence\sle_treg_case
set SEED=1729

call conda activate safefusion_env
if errorlevel 1 (echo FAILED: could not activate safefusion_env & exit /b 1)

for %%F in (1 2) do (
  set FOLD_DIR=%ROOT%\cite\fold_%%F
  set DEPLOY=!FOLD_DIR!\deploy

  echo.
  echo ================= FOLD %%F =================

  echo --- build: truth ---
  python scripts\sle_build_fold.py --prepared external_data\prepared\sle_cite.h5ad --splits !FOLD_DIR!\splits.parquet --output !FOLD_DIR!\truth.h5ad
  if errorlevel 1 (echo FAILED: build truth fold %%F & exit /b 1)

  echo --- build: mask ---
  python scripts\make_mask_replicate.py --truth !FOLD_DIR!\truth.h5ad --splits !FOLD_DIR!\splits.parquet --unit-column donor --seed %SEED% --output-dir !FOLD_DIR!\masked
  if errorlevel 1 (echo FAILED: mask fold %%F & exit /b 1)

  echo --- build: deploy/hybrid ---
  python scripts\build_deployment_inputs.py --truth !FOLD_DIR!\truth.h5ad --corrupted !FOLD_DIR!\masked\corrupted.h5ad --coordinates !FOLD_DIR!\masked\coordinates.parquet --splits !FOLD_DIR!\splits.parquet --output-dir !DEPLOY!
  if errorlevel 1 (echo FAILED: deploy fold %%F & exit /b 1)

  echo --- teachers: gene_median ---
  python scripts\run_leakage_safe_method.py --method gene_median --input !DEPLOY!\hybrid.h5ad --coordinates !DEPLOY!\coordinates.parquet --splits !DEPLOY!\splits.parquet --output !DEPLOY!\methods\gene_median --seed %SEED%
  if errorlevel 1 (echo FAILED: gene_median fold %%F & exit /b 1)

  echo --- teachers: svd_impute ---
  python scripts\run_leakage_safe_method.py --method svd_impute --input !DEPLOY!\hybrid.h5ad --coordinates !DEPLOY!\coordinates.parquet --splits !DEPLOY!\splits.parquet --output !DEPLOY!\methods\svd_impute --seed %SEED%
  if errorlevel 1 (echo FAILED: svd_impute fold %%F & exit /b 1)

  echo --- teachers: graph_smooth ---
  python scripts\run_leakage_safe_method.py --method graph_smooth --input !DEPLOY!\hybrid.h5ad --coordinates !DEPLOY!\coordinates.parquet --splits !DEPLOY!\splits.parquet --output !DEPLOY!\methods\graph_smooth --seed %SEED%
  if errorlevel 1 (echo FAILED: graph_smooth fold %%F & exit /b 1)

  echo --- teachers: scvi_inductive ^(safefusion_env^) ---
  python scripts\run_inductive_teacher.py --method scvi --input !DEPLOY!\hybrid.h5ad --coordinates !DEPLOY!\coordinates.parquet --splits !DEPLOY!\splits.parquet --output !DEPLOY!\methods\scvi_inductive --seed %SEED%
  if errorlevel 1 (echo FAILED: scvi_inductive fold %%F & exit /b 1)

  echo --- baseline: scvi ^(safefusion_env^) ---
  python scripts\run_scvi_baseline.py --corrupted !DEPLOY!\hybrid.h5ad --coordinates !DEPLOY!\coordinates.parquet --splits !DEPLOY!\splits.parquet --output !DEPLOY!\baselines\scvi --epochs 200 --seed %SEED%
  if errorlevel 1 (echo FAILED: scvi baseline fold %%F & exit /b 1)

  call conda activate safefusion_magic
  if errorlevel 1 (echo FAILED: could not activate safefusion_magic & exit /b 1)

  echo --- teachers: magic_inductive ^(safefusion_magic^) ---
  python scripts\run_inductive_teacher.py --method magic --input !DEPLOY!\hybrid.h5ad --coordinates !DEPLOY!\coordinates.parquet --splits !DEPLOY!\splits.parquet --output !DEPLOY!\methods\magic_inductive --seed %SEED% --n-jobs 8
  if errorlevel 1 (echo FAILED: magic_inductive fold %%F & exit /b 1)

  echo --- baseline: magic ^(safefusion_magic^) ---
  python scripts\run_magic_baseline.py --corrupted !DEPLOY!\hybrid.h5ad --coordinates !DEPLOY!\coordinates.parquet --splits !DEPLOY!\splits.parquet --output !DEPLOY!\baselines\magic --n-jobs 8 --seed %SEED%
  if errorlevel 1 (echo FAILED: magic baseline fold %%F & exit /b 1)

  call conda activate safefusion_env
  if errorlevel 1 (echo FAILED: could not re-activate safefusion_env & exit /b 1)

  echo --- stack: safe_fusion ---
  python scripts\run_leakage_safe_method.py --method safe_fusion --input !DEPLOY!\hybrid.h5ad --coordinates !DEPLOY!\coordinates.parquet --splits !DEPLOY!\splits.parquet --output !DEPLOY!\methods\safe_fusion --seed %SEED% --teacher-contract !DEPLOY!\methods\gene_median --teacher-contract !DEPLOY!\methods\svd_impute --teacher-contract !DEPLOY!\methods\graph_smooth --teacher-contract !DEPLOY!\methods\magic_inductive --teacher-contract !DEPLOY!\methods\scvi_inductive
  if errorlevel 1 (echo FAILED: safe_fusion stack fold %%F & exit /b 1)

  echo --- selector: scoring every gene ^(uses wrapper to avoid argv length limit^) ---
  python scripts\run_selector_all_genes.py --corrupted !DEPLOY!\hybrid.h5ad --truth !FOLD_DIR!\truth.h5ad --coordinates !DEPLOY!\coordinates.parquet --splits !DEPLOY!\splits.parquet --fusion-contract !DEPLOY!\methods\safe_fusion --teacher-contract !DEPLOY!\methods\gene_median --teacher-contract !DEPLOY!\methods\svd_impute --teacher-contract !DEPLOY!\methods\graph_smooth --teacher-contract !DEPLOY!\methods\magic_inductive --teacher-contract !DEPLOY!\methods\scvi_inductive --output-dir !DEPLOY!\selector --fit-split development --architecture mlp --budget-mode apply_topk --budgets 0.01 0.05 0.1 --curve-points 2 --seed %SEED%
  if errorlevel 1 (echo FAILED: selector fold %%F & exit /b 1)

  echo --- fills ---
  python scripts\sle_fills.py --set cite --fold %%F --app deploy
  if errorlevel 1 (echo FAILED: fills fold %%F & exit /b 1)

  echo === FOLD %%F COMPLETE ===
)

echo.
echo ============================================================
echo FOLDS 1 AND 2 COMPLETE. If fold 0 is also done, you can now run:
echo   python scripts\sle_evaluate_protein_multigene.py --stage select
echo ============================================================
endlocal
