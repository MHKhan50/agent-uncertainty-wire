# Correlated errors in multi-agent decisions

A synthetic experiment on how shared failures affect a team of three simulated agents.

## Research question

How does a growing chance of shared failure affect majority voting and confidence-weighted voting?

## Run (Python 3.9+, standard library only)

```bash
python3 run_experiment.py --seeds 5
python3 -m unittest discover -s tests -v
```

Outputs are written to `outputs/`.

## Design

- Three simulated agents with set skills 0.80, 0.70 and 0.62.
- Shared-shock probability q in {0, 0.25, 0.5, 0.75, 1}. q is the chance of the shared event in the simulator, not a measured correlation between agents.
- Three rules: majority vote, raw confidence-weighted vote, and confidence-weighted vote after a correction estimated on a separate calibration sample.
- Metrics: accuracy, Brier score and 10-bin expected calibration error (ECE). References are in the root `REFERENCES.md`.

## Limitations

Synthetic agents only; no language model is called. Five seeds are a small sample. The real-model follow-up is in `../real_llm_confidence`.
