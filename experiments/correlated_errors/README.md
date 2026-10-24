# Experiment 2 — Correlated Errors in Multi-Agent Decisions

A standalone, reproducible **synthetic** benchmark for studying how shared error shocks affect three-agent aggregation.

## Research question
How does increasing dependence in agent correctness affect majority voting versus confidence-weighted aggregation?

## Run (macOS; Python 3.9+; standard library only)
```bash
python3 run_experiment.py --seeds 10
python3 -m unittest discover -s tests -v
```
Outputs are written only to this package's `outputs/` directory. The runner does not edit another project.

## Design
- Three simulated agents with nominal skills 0.80, 0.70, 0.62.
- Shared-shock mixture probability q in {0, .25, .5, .75, 1}. **q is not a pairwise correlation coefficient.**
- Three policies: majority, raw-confidence weighted, and calibration-adjusted confidence weighted.
- Metrics: accuracy, Brier score, and ten-bin expected calibration error (ECE).
- Separate calibration and test samples per seed; correction is estimated from calibration data only.

## Caution
This is a synthetic methods demonstration, not an evaluation of real LLM agents or evidence of safety. Default five seeds are a smoke test; use 30–100+ seeds for more stable exploratory results, report uncertainty, and pre-specify primary outcomes.
