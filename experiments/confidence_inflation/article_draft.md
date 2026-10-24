# Confidence Inflation in Multi-Agent Decisions: A Synthetic Stress Test

## Abstract
Confidence-weighted aggregation may be affected when reported confidence is systematically inflated. This project implements a reproducible synthetic experiment that varies the inflation level and compares raw confidence weighting with a correction estimated on a separate calibration sample.

## Methods
Three simulated binary agents have nominal correctness probabilities 0.80, 0.70, and 0.62. The additive inflation parameter ranges from 0 to 0.25. Calibration and test scenarios use separate pseudo-random streams. The correction is estimated from calibration records only and applied to test confidence weights. Accuracy, Brier score, and ten-bin expected calibration error are recorded.

## Results
Run the included script and inspect `outputs/metrics_summary.csv` before inserting numeric findings. Do not report unexecuted results or imply that synthetic results establish behavior in deployed systems.

## Limitations and next steps
The generator does not model strategic misreporting, agent adaptation, communication, or distribution shift. Future experiments should compare additive correction with temperature scaling or isotonic calibration, ensure sufficient calibration data, and report uncertainty across many seeds.
