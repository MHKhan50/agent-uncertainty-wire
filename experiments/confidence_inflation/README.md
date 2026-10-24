# Experiment 3 — Confidence Inflation and Calibration

A standalone **synthetic** stress test for confidence-weighted agent aggregation under systematically inflated confidence reports.

## Research question
Does a simple calibration adjustment improve probability quality when simulated agents overstate confidence by a controllable amount?

## Run (macOS; Python 3.9+; standard library only)
```bash
python3 run_experiment.py --seeds 10
python3 -m unittest discover -s tests -v
```
Outputs go into this project's `outputs/` directory only.

## Design
Three agents have nominal skills 0.80, 0.70, and 0.62. Confidence inflation ranges from 0 to 0.25. A separate calibration sample estimates the mean reported-confidence minus correctness gap; the correction is then evaluated on held-out synthetic scenarios. Metrics are accuracy, Brier score, and ten-bin ECE.

## Limitations
This is not real LLM data. Additive correction is a deliberately simple baseline, not a universal calibration recommendation. Use more seeds and compare against well-established calibration methods before drawing conclusions.
