# Executed findings: RNA-seq analysis

Real public data from Monaco et al. (2019), not synthetic expression data. This project is a new secondary analysis of deposited TPM, not a reproduction of the complete original study.

## Observed results

The source matrix contains 127 samples and 58,311 gene IDs. Metadata filtering retains 112 samples, 28 immune-cell types and four donors. Out-of-donor accuracy is 0.92857, macro F1 is 0.92379 and dummy accuracy is 0.03571. Fold accuracies range from 0.89286 to 0.96429. The separate three-fold random-sample comparator scores 0.67857. The exact two-sided paired p-value floor is 0.125; no gene has BH q < 0.05.

Negative control: with the cell-type labels shuffled 20 times (class counts kept), the in-fold pipeline reaches a mean donor-held-out accuracy of 0.01071 (maximum 0.03571; chance is 0.03571). Choosing the 300 features on all samples before holding donors out, with the same shuffled labels, reaches a mean of 0.57098 (maximum 0.69643), which is the leak the in-fold design avoids.

## Scope of evidence

Only four donors contribute to the classification and paired contrast. These are healthy purified cell populations, not mixed clinical specimens. There is no independent study, disease cohort, prospective test or clinical validation. PCA is descriptive and uses all included samples; it is not part of validation. Grouped and random splits have different training sizes and class coverage, so their score difference does not isolate a leakage effect. In this dataset the random split performs worse. No hyperparameter search was performed. Exact sign-flip inference requires appropriate sign exchangeability under the null. TPM contrasts are not raw-count differential expression; no DESeq2/edgeR claims are made. BH guarantees depend on the relevant null dependence assumptions, and the small exact-p floor prevents discoveries here regardless.

The accompanying validation receipt identifies the analysis and dependency hashes and reports pytest and notebook execution. Re-run after changing code, sources or dependencies.
