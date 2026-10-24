# Confidence inflation and a one-number correction

A synthetic experiment on whether subtracting a measured confidence gap helps when agents overstate their certainty.

## Research question

If agents add a fixed amount to their confidence, how much does a single correction, estimated on a separate sample, improve Brier score and calibration?

## Run (Python 3.9+, standard library only)

```bash
python3 run_experiment.py --seeds 5
python3 -m unittest discover -s tests -v
```

Outputs are written to `outputs/`.

## Design

- Three simulated agents with set skills 0.80, 0.70 and 0.62.
- Inflation levels 0, 0.05, 0.10, 0.15, 0.20 and 0.25.
- Per seed: 1,000 calibration scenarios to estimate the gap and 2,000 held-out scenarios to test it.
- Metrics: accuracy, Brier score and 10-bin expected calibration error (ECE). References are in the root `REFERENCES.md`.

## Limitations

Synthetic agents only; no language model is called. The correction is deliberately the simplest one and is not yet compared with standard methods such as temperature scaling.
