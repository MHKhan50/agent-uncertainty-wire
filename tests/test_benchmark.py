import copy
import unittest
from src.uncertainty_benchmark import POLICIES, REGIMES, calibration_offsets, evaluate_dataset, generate_dataset, summarize_seeds, agent_calibration_summary, expected_calibration_error, _team_probability, _logit, _sigmoid, agent_reliability_bins, paired_policy_differences

class BenchmarkTests(unittest.TestCase):
    def test_dataset_shape_splits_and_confidence_bounds(self):
        rows = generate_dataset(scenarios_per_seed=20, seeds=(7,))
        self.assertEqual(len(rows), 20 * 3 * 2)
        self.assertEqual(set(r['regime'] for r in rows), set(REGIMES))
        self.assertTrue(all(0.51 <= r['reported_confidence'] <= 0.99 for r in rows))
        for regime in REGIMES:
            group = [r for r in rows if r['regime'] == regime]
            self.assertEqual(sum(r['split'] == 'calibration' for r in group), 8 * 3)
            self.assertEqual(sum(r['split'] == 'test' for r in group), 12 * 3)

    def test_calibration_uses_no_test_rows(self):
        rows = generate_dataset(scenarios_per_seed=30, seeds=(11,))
        original = calibration_offsets(rows)
        changed = copy.deepcopy(rows)
        for row in changed:
            if row['split'] == 'test':
                row['reported_confidence'] = 0.99
                row['actual_correct'] = 1 - int(row['actual_correct'])
        self.assertEqual(original, calibration_offsets(changed))

    def test_metrics_are_bounded_and_only_test_predictions_are_scored(self):
        rows = generate_dataset(scenarios_per_seed=50, seeds=(13, 19))
        seed_summary, predictions = evaluate_dataset(rows)
        self.assertEqual(len(seed_summary), 2 * len(REGIMES) * len(POLICIES))
        self.assertTrue(all(r['split'] == 'test' for r in predictions))
        self.assertTrue(all(0 <= r['accuracy'] <= 1 for r in seed_summary))
        self.assertTrue(all(0 <= r['brier'] <= 1 for r in seed_summary))
        self.assertTrue(all(0 <= r['ece'] <= 1 for r in seed_summary))
        self.assertEqual(len(summarize_seeds(seed_summary)), len(REGIMES) * len(POLICIES))


    def test_agent_calibration_diagnostics_are_held_out_and_bounded(self):
        rows = generate_dataset(scenarios_per_seed=100, seeds=(31,))
        diagnostics = agent_calibration_summary(rows)
        self.assertEqual(len(diagnostics), len(REGIMES) * 3)
        for row in diagnostics:
            self.assertEqual(row['n'], 60)
            for key in ('raw_brier_correctness', 'calibrated_brier_correctness', 'raw_ece_correctness', 'calibrated_ece_correctness'):
                self.assertTrue(0 <= row[key] <= 1)

    def test_ece_known_case(self):
        # One perfectly calibrated observation in each bin has zero ECE.
        self.assertAlmostEqual(expected_calibration_error([0.0, 1.0], [0, 1]), 0.0)
        # Two observations at confidence .8 with empirical accuracy .5 yield .3 ECE.
        self.assertAlmostEqual(expected_calibration_error([0.8, 0.8], [0, 1]), 0.3)

    def test_independent_log_odds_aggregation_known_cases(self):
        rows = [
            {'seed': 1, 'regime': 'calibrated', 'agent_id': 'a', 'predicted_label': 1, 'reported_confidence': 0.8},
            {'seed': 1, 'regime': 'calibrated', 'agent_id': 'b', 'predicted_label': 0, 'reported_confidence': 0.8},
        ]
        # Equally reliable opposing agents cancel to the balanced prior.
        self.assertAlmostEqual(_team_probability(rows, 'independent_log_odds_raw', {}), 0.5, places=7)
        rows[1]['predicted_label'] = 1
        expected = _sigmoid(2 * _logit(0.8))
        self.assertAlmostEqual(_team_probability(rows, 'independent_log_odds_raw', {}), expected, places=7)

    def test_reliability_and_paired_outputs_are_well_formed(self):
        rows = generate_dataset(scenarios_per_seed=100, seeds=(17, 29, 43, 71, 101))
        reliability = agent_reliability_bins(rows)
        self.assertTrue(reliability)
        self.assertTrue(all(0 <= r['observed_accuracy'] <= 1 for r in reliability))
        seed_summary, _ = evaluate_dataset(rows)
        paired = paired_policy_differences(seed_summary)
        self.assertTrue(paired)
        self.assertTrue(all(r['ci95_low'] <= r['ci95_high'] for r in paired))

    def test_generation_is_deterministic(self):
        self.assertEqual(generate_dataset(12, (23,)), generate_dataset(12, (23,)))

if __name__ == '__main__':
    unittest.main()
