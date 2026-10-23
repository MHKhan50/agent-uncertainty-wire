# Agent Uncertainty Wire: Comparing Confidence Aggregation Rules in a Synthetic Multi-Agent Task

**Author:** Muzammal Hussain  
**Artifact version:** 0.3.0  
**Date:** 23 October 2026

> **Status:** Reproducible synthetic benchmark. This draft is not peer reviewed and does not evaluate real LLM agents.

## Abstract

Multi-agent systems may pass confidence scores alongside predicted labels, but reported confidence can be miscalibrated and aggregation rules can affect downstream probability quality. This synthetic benchmark compares five team-level policies: label-only majority voting, two arithmetic confidence-weighted baselines, and two independent log-odds aggregators. A per-agent logistic calibrator is fitted using a separate calibration split. Five seeds and two paired confidence regimes are evaluated using accuracy, Brier score and expected calibration error (ECE). Additional held-out agent-level diagnostics and reliability diagrams assess calibration separately from team-level performance. The results are simulator-dependent and do not establish superiority for real agent systems.

## Research question

How do confidence aggregation rules compare in a controlled synthetic binary task, and does correcting agent-level confidence necessarily improve team-level predictions?

## Motivation and scope

The University of Edinburgh’s official uncertainty-aware agentic-AI theme identifies uncertainty elicitation, communication, propagation and team-level calibration as research objectives. The University of Glasgow’s 2026 agentic-AI/software-engineering partnership and the University of Surrey’s announced AI-agent containment project are related context; they do not establish novelty or endorse this work. See `REFERENCES.md`.

## Methods

The default run generates 2,500 scenarios for each of five seeds. Each scenario has three simulated agents with different nominal skills. Forty per cent of scenario IDs are reserved for calibration and sixty per cent are held out. Each base scenario is paired across `calibrated` and `overconfident` confidence regimes; the latter adds 0.18 to reported confidence before clipping.

The policies are: (1) `label_only_majority`, which averages binary labels and thresholds at 0.5; (2) `confidence_weighted_raw`, which averages class probabilities implied by reported confidence; (3) `confidence_weighted_calibrated`, which uses a per-agent logistic mapping fitted to calibration rows only; (4) `independent_log_odds_raw`; and (5) `independent_log_odds_calibrated`. The log-odds rules combine signed log-odds of each agent’s probability of being correct, under a balanced prior and a conditional-independence assumption. This assumption may fail when agent errors are correlated, so the log-odds rules are treated as explicit comparison baselines rather than universally valid aggregators.

Metrics are computed on held-out test scenarios. Accuracy is higher-is-better; Brier score and ten-bin ECE are lower-is-better. Values are means across five seeds; ± is between-seed sample standard deviation, not a confidence interval. The separate paired-difference file reports descriptive t intervals across seeds; with only five seeds, those intervals are imprecise.

## Team-level results

| Confidence regime | Policy | Test predictions | Accuracy (mean ± SD) | Brier (mean ± SD) | ECE (mean ± SD) |
|---|---|---:|---:|---:|---:|
| calibrated | `confidence_weighted_calibrated` | 7500 | 0.731 ± 0.013 | 0.206 ± 0.003 | 0.146 ± 0.012 |
| calibrated | `confidence_weighted_raw` | 7500 | 0.733 ± 0.013 | 0.205 ± 0.002 | 0.142 ± 0.013 |
| calibrated | `independent_log_odds_calibrated` | 7500 | 0.733 ± 0.013 | 0.179 ± 0.005 | 0.018 ± 0.005 |
| calibrated | `independent_log_odds_raw` | 7500 | 0.736 ± 0.014 | 0.180 ± 0.006 | 0.019 ± 0.009 |
| calibrated | `label_only_majority` | 7500 | 0.717 ± 0.013 | 0.198 ± 0.004 | 0.055 ± 0.009 |
| overconfident | `confidence_weighted_calibrated` | 7500 | 0.733 ± 0.012 | 0.207 ± 0.003 | 0.149 ± 0.011 |
| overconfident | `confidence_weighted_raw` | 7500 | 0.718 ± 0.010 | 0.185 ± 0.004 | 0.064 ± 0.009 |
| overconfident | `independent_log_odds_calibrated` | 7500 | 0.733 ± 0.014 | 0.180 ± 0.005 | 0.017 ± 0.004 |
| overconfident | `independent_log_odds_raw` | 7500 | 0.729 ± 0.012 | 0.206 ± 0.009 | 0.152 ± 0.012 |
| overconfident | `label_only_majority` | 7500 | 0.717 ± 0.013 | 0.198 ± 0.004 | 0.055 ± 0.009 |

## Agent-level calibration diagnostics

The following values average seed-agent groups within each regime. They measure the predicted probability that an individual agent's label is correct, not the team's probability of class 1.

| Regime | Raw Brier | Calibrated Brier | Raw ECE | Calibrated ECE |
|---|---:|---:|---:|---:|
| calibrated | 0.2230 | 0.2227 | 0.0224 | 0.0191 |
| overconfident | 0.2553 | 0.2229 | 0.1801 | 0.0228 |

The calibration model reduces the imposed overconfidence penalty in these synthetic agent-level diagnostics. However, team-level results depend on aggregation assumptions and the simulator's error structure. Agent-level calibration and team-level calibration are different objectives.

## Figures and generated files

- `outputs/accuracy.svg` — held-out team accuracy by regime and policy.
- `outputs/brier_score.svg` — held-out team Brier score.
- `outputs/ece.svg` — held-out team ECE.
- `outputs/agent_reliability_calibrated.svg` and `outputs/agent_reliability_overconfident.svg` — raw and calibrated agent-level reliability diagrams.
- `outputs/paired_policy_differences.csv` — paired policy differences with descriptive seed-level intervals.
- `outputs/agent_calibration_diagnostics.csv` and `outputs/agent_reliability_bins.csv` — agent-level diagnostics.

## Interpretation

The benchmark does not establish that one aggregation policy is universally best. Arithmetic averaging and log-odds aggregation answer different modelling assumptions; the log-odds method is sensitive to its conditional-independence assumption. ECE depends on binning and can hide subgroup miscalibration. Accuracy, Brier score and ECE should be read together, and small differences across only five seeds should be interpreted cautiously. The experiment is not evidence of real-world safety, generalisation or the desirability of overconfidence in real agents.

## Reproducibility

Run `python3 run_all.py` followed by `python3 -m unittest discover -s tests -v`. The implementation uses the Python standard library only. The default run regenerates synthetic data, 75,000 held-out predictions, CSV diagnostics, paired seed-level intervals and SVG graphs. Eight automated tests pass in the package validation run. No LLM was called.

## Limitations

This is a synthetic binary task with three simulated agents and five seeds. The simulator’s ground-truth probability field is not available in a real system. ECE depends on binning and is not a complete calibration diagnosis. The experiment does not model natural-language messages, tool use, long-horizon planning, adversarial agents, real-world distribution shift, human oversight or production latency. Log-odds aggregation assumes conditionally independent agent errors and a balanced prior; these assumptions may be unrealistic. No real-world safety or generalisation claim is made.

## References

1. University of Edinburgh, “Uncertainty-Aware Agentic AI”: https://eng.ed.ac.uk/spads-cdt/studying/degrees/postgraduate-research/phd/uncertainty-aware-agentic-ai
2. University of Glasgow, agentic AI and software/data engineering partnership (30 March 2026): https://www.gla.ac.uk/news/headline_1256583_en.html
3. University of Surrey, secure sandbox for AI agents (15 October 2026): https://www.surrey.ac.uk/news/surrey-leads-ps32m-project-collaboration-sheffield-and-melbourne-develop-provably-secure-sandbox-ai
4. Guo et al. (2017), “On Calibration of Modern Neural Networks,” PMLR: https://proceedings.mlr.press/v70/guo17a.html
5. Brier (1950), “Verification of Forecasts Expressed in Terms of Probability,” Monthly Weather Review: https://doi.org/10.1175/1520-0493%281950%29078%3C0001%3AVOFEIT%3E2.0.CO%3B2

## Citation

Muzammal Hussain (2026). *Agent Uncertainty Wire: a reproducible synthetic benchmark*, version 0.3.0. Cite a versioned repository or release when available; no DOI has been minted.
