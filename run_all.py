#!/usr/bin/env python3
"""Regenerate synthetic data, held-out metrics and SVG graphs."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.uncertainty_benchmark import DEFAULT_SEEDS, DEFAULT_SCENARIOS_PER_SEED, generate_dataset, evaluate_dataset, summarize_seeds, write_csv, plot_metric, agent_calibration_summary, paired_policy_differences, agent_reliability_bins, plot_agent_reliability
DATA, OUT = ROOT / 'data', ROOT / 'outputs'
DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
print('Generating deterministic synthetic agent messages...')
rows = generate_dataset()
write_csv(DATA / 'agent_messages.csv', rows, ['seed','scenario_id','split','regime','truth','difficulty','agent_id','predicted_label','reported_confidence','generator_accuracy_probability','actual_correct'])
print('Evaluating held-out scenarios; calibration rows are excluded from test metrics...')
seed_summary, predictions = evaluate_dataset(rows)
summary = summarize_seeds(seed_summary)
write_csv(OUT / 'predictions.csv', predictions, ['seed','scenario_id','regime','split','truth','policy','probability_class_1','predicted_label','confidence','correct','brier'])
write_csv(OUT / 'seed_summary.csv', seed_summary, ['seed','regime','policy','n','accuracy','brier','ece'])
write_csv(OUT / 'metrics_summary.csv', summary, ['regime','policy','n','accuracy','accuracy_sd_across_seeds','brier','brier_sd_across_seeds','ece','ece_sd_across_seeds'])
write_csv(OUT / 'paired_policy_differences.csv', paired_policy_differences(seed_summary), ['regime','metric','policy_a','policy_b','n_paired_seeds','mean_difference_a_minus_b','ci95_low','ci95_high','interval_method'])
plot_metric(summary,'accuracy','Held-out team accuracy',str(OUT/'accuracy.svg'),1.0)
plot_metric(summary,'brier','Held-out Brier score',str(OUT/'brier_score.svg'),0.5)
plot_metric(summary,'ece','Held-out expected calibration error',str(OUT/'ece.svg'),0.5)
agent_diag = agent_calibration_summary(rows)
write_csv(OUT / 'agent_calibration_diagnostics.csv', agent_diag, ['seed','regime','agent_id','n','raw_brier_correctness','calibrated_brier_correctness','raw_ece_correctness','calibrated_ece_correctness'])
reliability = agent_reliability_bins(rows)
write_csv(OUT / 'agent_reliability_bins.csv', reliability, ['regime','method','bin','bin_lower','bin_upper','n','mean_confidence','observed_accuracy'])
for regime in ('calibrated','overconfident'):
    plot_agent_reliability(reliability, regime, str(OUT / ('agent_reliability_' + regime + '.svg')))
print('Seeds: {}'.format(', '.join(map(str, DEFAULT_SEEDS))) )
print('Scenarios per seed: {}'.format(DEFAULT_SCENARIOS_PER_SEED))
print('Agent-message rows: {}'.format(len(rows)))
print('Held-out predictions: {}'.format(len(predictions)))
print('Metrics, paired seed-level intervals, reliability bins and calibration diagnostics written to outputs/. No LLM calls were made.')
