# Correlated Errors in Multi-Agent Decisions: A Reproducible Synthetic Benchmark

## Abstract
Multi-agent systems may combine several recommendations without accounting for dependence between their errors. This package provides a reproducible synthetic experiment that varies a shared-error mechanism and compares majority voting with confidence-weighted aggregation. It reports accuracy, Brier score, and expected calibration error across dependence settings.

## Methods
Three simulated binary agents have nominal correctness probabilities 0.80, 0.70, and 0.62. The shared-shock mixture probability q ranges from 0 to 1. For each scenario, a shared draw determines correctness with probability q; otherwise agent correctness draws are independent. The q parameter is a design mixture probability, not a measured pairwise correlation. A separate calibration sample estimates an additive confidence correction, which is applied to test-sample confidence reports. All results are generated synthetically.

## Results
Run the included experiment and inspect `outputs/metrics_summary.csv` before inserting numerical findings. Do not claim results that have not been checked. The default five seeds are exploratory only.

## Limitations
The shared-shock process is only one dependence model. The benchmark does not evaluate deployed systems, strategic behavior, communication effects, or distribution shift. Future work should vary the dependence structure, agent skills, calibration method, and number of seeds.

## Reproducibility
The runner, tests, data card, generated output, and manifest are included in this package.
