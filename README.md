# **SafeFusion Validation Research**

Independent validation and debugging of **SafeFusion**, a machine-learning zero-imputation method for single-cell RNA-seq data, benchmarked against **scVI**. I built the test harness, caught a critical data-integrity bug, and stress-tested results against statistical confounders.

*Contribution to an ongoing research project (paper in progress). The validation and debugging work is mine; the model-training code is the lab's existing codebase.*

---

## **The problem**

Single-cell sequencing produces many *false zeros* (a gene was active but the reading missed it). Imputation fills these in, but must not corrupt *genuine* biological zeros. The two look identical in the data, so validation requires an **independent source of truth**. I built three such tests on a 20,000-cell CRISPR knockout dataset.

## **What I did**

* **Built an end-to-end validation pipeline** (Python, pandas, NumPy, scikit-learn) to benchmark the method against a deep-learning baseline across \~20,000 cells and \~18,000 genes, using CRISPR knockouts as biological ground truth.  
* **Caught and fixed a critical bug:** a validation test was scoring data against itself because its ground-truth file had been generated from the labels it was meant to check. I traced the root cause, rebuilt the test from real measurement data, and verified the fix three independent ways: 100% label agreement on 20,156 records, correct feature mapping, and reproduction of a known published biological result.  
* **Ran a confounder-controlled analysis** across \~21,000 features to test whether a promising signal was real or an artifact, confirming it survived sequencing-depth and feature-identity controls. Then I \`reported it honestly as modest supporting evidence rather than overstating it.

## **Selected results**

| Test | Metric  | scVI | safeFusion |
| :---- | :---- | :---- | :---- |
| On-target Precision | precision@ⲧ | 0.613 | 0.712 |
| Downstream signal | logFC correlation | 0.804 | 0.995 |
| Cross-modal (rebuilt) | ROC-AUC | 0.727 | 0.732 |

*Overall Report: SafeFusion is the most precise imputer and best preserves downstream signal, with modest gains over scVI. Limitations are documented in-repo.*

## **Stack**

Python · pandas · NumPy · SciPy · scikit-learn · scanpy · anndata · scvi-tools · Git

## **Status**

Work in progress. **Done:** validation harness, cross-modal test rebuild, confounder analysis. **Next:** certifying knockout labels, remaining confounder checks, and integrating multiple evidence sources into one framework.

