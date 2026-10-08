@echo off
REM ============================================================
REM Builds the Hao 2021 CITE-seq ("cite") fold pipeline needed by
REM sle_evaluate_protein_multigene.py: prepare -> build -> teachers
REM -> scVI/MAGIC -> stack (Safe Fusion) -> selector -> fills,
REM for 3 donor-held-out folds. De-arrayed from
REM scripts\slurm_sle_pipeline.sh (STAGE=prepare/build/teachers/
REM scvi_teacher/magic/scvi/stack/selector/fills), restricted to
REM the cite set only.
REM
REM Run this from an Anaconda Prompt, from the repo root:
REM   D:\bioinformatics_research\codebases\safe-fusion-stn> build_cite_pipeline.bat
REM
REM Requires two conda envs to already exist:
REM   safefusion_env    (core + scVI -- you already have this)
REM   safefusion_magic  (MAGIC only -- created separately)
REM
REM Requires external_data\citeseq_hao2021\ to already hold the
REM 7 GEO files (see fetch step in chat).
REM ============================================================

setlocal enabledelayedexpansion

set ROOT=artifacts\paper_evidence\sle_treg_case
set CAP=100
set SEED=1729

echo.
echo === Checking raw Hao data is present ===
if not exist external_data\citeseq_hao2021\GSM5008737_RNA_3P-matrix.mtx.gz (
  echo FAILED: external_data\citeseq_hao2021 is missing the GEO files. Fetch them first.
  exit /b 1
)

call conda activate safefusion_env
if errorlevel 1 (echo FAILED: could not activate safefusion_env & exit /b 1)

echo.
echo === Stage: prepare (cite, once) ===
python scripts\sle_prepare_citeseq.py --output external_data\prepared\sle_cite.h5ad --adt-output %ROOT%\cite\adt.parquet --fold-dir %ROOT%\cite --cap-per-type %CAP% --folds 3 --seed %SEED%
if errorlevel 1 (echo FAILED: prepare & exit /b 1)

for %%F in (0 1 2) do (
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
echo ALL FOLDS COMPLETE. You can now run:
echo   python scripts\sle_evaluate_protein_multigene.py --stage select
echo ============================================================
endlocal
