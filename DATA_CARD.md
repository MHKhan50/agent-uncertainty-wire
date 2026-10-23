# Data card: synthetic agent uncertainty messages

## Summary

`data/agent_messages.csv` is generated entirely by `src/uncertainty_benchmark.py`. It contains no personal data and is not derived from a third-party dataset. The default configuration has five seeds, 2,500 scenarios per seed, three simulated agents and two paired confidence regimes.

## Intended use

- Teaching reproducible experimental design and held-out evaluation.
- Comparing simple aggregation baselines under controlled assumptions.
- Demonstrating accuracy, Brier score and expected calibration error.
- Prototyping data processing and graph-generation workflows.

## Out of scope

- Estimating reliability of real LLMs or production agents.
- Demonstrating safety, security or generalisation to real environments.
- Making clinical, financial, legal or other high-stakes decisions.
- Treating simulator ground-truth metadata as available at inference time.

## Fields

| Field | Meaning |
|---|---|
| `seed` | Random seed for the generated scenario set. |
| `scenario_id` | Scenario index within a seed. |
| `split` | `calibration` or held-out `test`. |
| `regime` | `calibrated` or deliberately `overconfident` reported confidence. |
| `truth` | Synthetic binary ground-truth label (0 or 1). |
| `difficulty` | Synthetic scenario difficulty value. |
| `agent_id` | One of three simulated agents with different nominal skills. |
| `predicted_label` | Agent's binary prediction. |
| `reported_confidence` | Communicated confidence, clipped to [0.51, 0.99]. |
| `generator_accuracy_probability` | Simulator's probability that this agent is correct; oracle metadata, not available in a real deployment. |
| `actual_correct` | Whether the simulated prediction matches truth. |

## Generation and limitations

For each scenario, truth, difficulty and agent predictions are generated once and reused across both regimes. The overconfident regime adds 0.18 to reported confidence before clipping, isolating a confidence-reporting intervention while keeping predictions paired. Calibration corrections use only calibration rows for the matching seed, regime and agent. The hand-designed skills, confidence noise and binary task are arbitrary; correlated real-world errors are not faithfully modelled.

## Licence and provenance

Generated data: CC BY 4.0 (`DATA_LICENSE.txt`). Code: MIT (`LICENSE`). No third-party dataset is bundled. References in `REFERENCES.md` informed topic selection and metric framing; they are not data sources.
