"""Dependency-free synthetic benchmark for confidence-aware agent handoffs."""
import random
import statistics
import math
from pathlib import Path
from collections import defaultdict

AGENT_SKILLS = {"agent_alpha": 0.80, "agent_beta": 0.70, "agent_gamma": 0.62}
REGIMES = {"calibrated": 0.0, "overconfident": 0.18}
POLICIES = ("label_only_majority", "confidence_weighted_raw", "confidence_weighted_calibrated", "independent_log_odds_raw", "independent_log_odds_calibrated")
DEFAULT_SEEDS = (17, 29, 43, 71, 101)
DEFAULT_SCENARIOS_PER_SEED = 2500
CALIBRATION_FRACTION = 0.40


def clamp(value, low, high):
    return max(low, min(high, value))


def generate_dataset(scenarios_per_seed=DEFAULT_SCENARIOS_PER_SEED, seeds=DEFAULT_SEEDS):
    """Generate long-form synthetic messages; regimes share truth and predictions."""
    if scenarios_per_seed < 10:
        raise ValueError("Use at least 10 scenarios per seed so both splits are meaningful.")
    rows = []
    calibration_count = int(scenarios_per_seed * CALIBRATION_FRACTION)
    for seed in seeds:
        rng = random.Random(seed)
        for scenario_id in range(scenarios_per_seed):
            split = "calibration" if scenario_id < calibration_count else "test"
            truth = rng.randint(0, 1)
            difficulty = rng.uniform(0.0, 0.12)
            messages = []
            for agent_id, nominal_skill in AGENT_SKILLS.items():
                p_correct = clamp(nominal_skill - difficulty + rng.uniform(-0.03, 0.03), 0.51, 0.95)
                correct = int(rng.random() < p_correct)
                predicted = truth if correct else 1 - truth
                base_conf = clamp(p_correct + rng.uniform(-0.07, 0.07), 0.51, 0.94)
                messages.append((agent_id, predicted, p_correct, base_conf, correct))
            for regime, shift in REGIMES.items():
                for agent_id, predicted, p_correct, base_conf, correct in messages:
                    rows.append({
                        "seed": seed, "scenario_id": scenario_id, "split": split,
                        "regime": regime, "truth": truth, "difficulty": round(difficulty, 6),
                        "agent_id": agent_id, "predicted_label": predicted,
                        "reported_confidence": round(clamp(base_conf + shift, 0.51, 0.99), 6),
                        "generator_accuracy_probability": round(p_correct, 6),
                        "actual_correct": correct,
                    })
    return rows


def _logit(p):
    p = clamp(float(p), 1e-5, 1.0 - 1e-5)
    return math.log(p / (1.0 - p))


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-min(x, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -60.0))
    return z / (1.0 + z)


def fit_logistic_calibrators(rows):
    """Fit per-seed/regime/agent Platt calibrators on calibration rows only.

    Target is whether the agent's stated label was correct. The sole feature is
    logit(reported confidence). A small L2 penalty stabilises the slope.
    """
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] == "calibration":
            key = (int(row["seed"]), row["regime"], row["agent_id"])
            grouped[key].append((_logit(float(row["reported_confidence"])), int(row["actual_correct"])))
    models = {}
    for key, values in grouped.items():
        a, b = 0.0, 1.0  # intercept and slope
        # Mean-loss gradient descent avoids ill-conditioned 2x2 Hessians when
        # reported confidences occupy a narrow range.
        n = float(len(values))
        learning_rate = 0.05
        for _ in range(400):
            g0 = g1 = 0.0
            for x, y in values:
                error = _sigmoid(a + b*x) - y
                g0 += error
                g1 += error*x
            g0 /= n
            g1 = g1 / n + 0.01*b  # L2 regularisation on slope
            da, db = learning_rate*g0, learning_rate*g1
            a -= da
            b -= db
            if abs(da) + abs(db) < 1e-8:
                break
        models[key] = (a, b)
    return models


def calibration_offsets(rows):
    """Backward-compatible alias; v0.2 returns fitted logistic parameters."""
    return fit_logistic_calibrators(rows)


def calibrated_correctness_probability(row, models):
    key = (int(row["seed"]), row["regime"], row["agent_id"])
    a, b = models[key]
    return clamp(_sigmoid(a + b*_logit(float(row["reported_confidence"]))), 0.50, 0.99)

def expected_calibration_error(confidences, correct_flags, bins=10):
    if not confidences:
        return 0.0
    grouped = [[] for _ in range(bins)]
    for confidence, correct in zip(confidences, correct_flags):
        grouped[min(bins - 1, int(clamp(confidence, 0.0, 1.0) * bins))].append((confidence, correct))
    total = len(confidences)
    return sum((len(group) / total) * abs(statistics.mean(x[0] for x in group) - statistics.mean(x[1] for x in group))
               for group in grouped if group)


def _team_probability(agent_rows, policy, calibrators):
    """Return P(team truth=1) under an explicitly named aggregation rule.

    The arithmetic-average policies are simple soft-voting baselines. The
    independent-log-odds policies combine each agent's correctness probability
    as a likelihood ratio under a balanced prior and conditional-independence
    assumption. That assumption is deliberately explicit and may fail when
    agents share data, models, prompts or failure modes.
    """
    if policy == "label_only_majority":
        return statistics.mean(int(row["predicted_label"]) for row in agent_rows)
    is_calibrated = policy in ("confidence_weighted_calibrated", "independent_log_odds_calibrated")
    is_log_odds = policy in ("independent_log_odds_raw", "independent_log_odds_calibrated")
    if policy not in POLICIES:
        raise ValueError("Unknown policy: {}".format(policy))
    probabilities = []
    log_odds = 0.0
    for row in agent_rows:
        confidence = (calibrated_correctness_probability(row, calibrators) if is_calibrated
                      else float(row["reported_confidence"]))
        confidence = clamp(confidence, 0.500001, 0.999999)
        predicted = int(row["predicted_label"])
        if is_log_odds:
            signed_log_odds = _logit(confidence) if predicted == 1 else -_logit(confidence)
            log_odds += signed_log_odds
        else:
            probabilities.append(confidence if predicted == 1 else 1.0 - confidence)
    return _sigmoid(log_odds) if is_log_odds else statistics.mean(probabilities)


def evaluate_dataset(rows):
    offsets = calibration_offsets(rows)
    scenarios = defaultdict(list)
    for row in rows:
        if row["split"] == "test":
            scenarios[(int(row["seed"]), row["regime"], int(row["scenario_id"]))].append(row)
    predictions = []
    grouped = defaultdict(list)
    for (seed, regime, scenario_id), agent_rows in sorted(scenarios.items()):
        agent_rows = sorted(agent_rows, key=lambda row: row["agent_id"])
        truth = int(agent_rows[0]["truth"])
        for policy in POLICIES:
            p1 = _team_probability(agent_rows, policy, offsets)
            predicted = int(p1 >= 0.5)
            confidence = max(p1, 1.0 - p1)
            result = {"seed": seed, "scenario_id": scenario_id, "regime": regime, "split": "test",
                      "truth": truth, "policy": policy, "probability_class_1": round(p1, 8),
                      "predicted_label": predicted, "confidence": round(confidence, 8),
                      "correct": int(predicted == truth), "brier": round((p1 - truth) ** 2, 8)}
            predictions.append(result)
            grouped[(seed, regime, policy)].append(result)
    seed_summary = []
    for (seed, regime, policy), values in sorted(grouped.items()):
        seed_summary.append({"seed": seed, "regime": regime, "policy": policy, "n": len(values),
                             "accuracy": statistics.mean(x["correct"] for x in values),
                             "brier": statistics.mean(x["brier"] for x in values),
                             "ece": expected_calibration_error([x["confidence"] for x in values], [x["correct"] for x in values])})
    return seed_summary, predictions



def agent_calibration_summary(rows):
    """Held-out agent-level calibration diagnostics, before and after Platt scaling."""
    models = fit_logistic_calibrators(rows)
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] != "test":
            continue
        key = (int(row["seed"]), row["regime"], row["agent_id"])
        grouped[key].append(row)
    result = []
    for (seed, regime, agent_id), values in sorted(grouped.items()):
        raw = [float(r["reported_confidence"]) for r in values]
        calibrated = [calibrated_correctness_probability(r, models) for r in values]
        correct = [int(r["actual_correct"]) for r in values]
        result.append({
            "seed": seed, "regime": regime, "agent_id": agent_id, "n": len(values),
            "raw_brier_correctness": statistics.mean((p-y)**2 for p,y in zip(raw,correct)),
            "calibrated_brier_correctness": statistics.mean((p-y)**2 for p,y in zip(calibrated,correct)),
            "raw_ece_correctness": expected_calibration_error(raw, correct),
            "calibrated_ece_correctness": expected_calibration_error(calibrated, correct),
        })
    return result

def paired_policy_differences(seed_summary):
    """Paired 95% t intervals across seeds for policy metric differences.

    Difference is policy_a minus policy_b. Intervals use the two-sided Student-t
    critical value for the observed degrees of freedom; this is descriptive with
    only five default seeds and should not be overinterpreted.
    """
    t_critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                  6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                  11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
                  16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
                  21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
                  26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}
    lookup = {(int(r['seed']), r['regime'], r['policy']): r for r in seed_summary}
    output = []
    for regime in REGIMES:
        for metric in ('accuracy', 'brier', 'ece'):
            for i, policy_a in enumerate(POLICIES):
                for policy_b in POLICIES[i+1:]:
                    diffs = [float(lookup[(seed, regime, policy_a)][metric]) -
                             float(lookup[(seed, regime, policy_b)][metric])
                             for seed in DEFAULT_SEEDS if (seed, regime, policy_a) in lookup and (seed, regime, policy_b) in lookup]
                    if len(diffs) < 2:
                        continue
                    mean_diff = statistics.mean(diffs)
                    se = statistics.stdev(diffs) / math.sqrt(len(diffs))
                    critical = t_critical.get(len(diffs)-1, 1.96)
                    output.append({'regime': regime, 'metric': metric, 'policy_a': policy_a,
                                   'policy_b': policy_b, 'n_paired_seeds': len(diffs),
                                   'mean_difference_a_minus_b': mean_diff,
                                   'ci95_low': mean_diff-critical*se, 'ci95_high': mean_diff+critical*se,
                                   'interval_method': 'paired t interval across seeds; descriptive'})
    return output


def agent_reliability_bins(rows, bins=10):
    """Held-out reliability-bin data for raw and calibrated agent confidence."""
    models = fit_logistic_calibrators(rows)
    grouped = defaultdict(list)
    for row in rows:
        if row['split'] != 'test':
            continue
        key = (row['regime'], 'raw')
        grouped[key].append((float(row['reported_confidence']), int(row['actual_correct'])))
        key = (row['regime'], 'calibrated')
        grouped[key].append((calibrated_correctness_probability(row, models), int(row['actual_correct'])))
    output = []
    for (regime, method), values in sorted(grouped.items()):
        for bin_index in range(bins):
            lower, upper = bin_index / bins, (bin_index + 1) / bins
            chunk = [(p, y) for p, y in values if (lower <= p < upper or (bin_index == bins-1 and p == 1.0))]
            if not chunk:
                continue
            output.append({'regime': regime, 'method': method, 'bin': bin_index,
                           'bin_lower': lower, 'bin_upper': upper, 'n': len(chunk),
                           'mean_confidence': statistics.mean(p for p, _ in chunk),
                           'observed_accuracy': statistics.mean(y for _, y in chunk)})
    return output


def plot_agent_reliability(reliability_rows, regime, filename):
    """Write a compact reliability diagram for held-out agent correctness."""
    width, height, left, right, top, bottom = 720, 620, 90, 45, 55, 90
    plot_w, plot_h = width-left-right, height-top-bottom
    def xy(x, y):
        return left + x*plot_w, top + (1-y)*plot_h
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(width,height,width,height),
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="{}" y="28" font-family="Arial" font-size="20" font-weight="bold">Agent reliability: {} regime</text>'.format(left, regime)]
    for tick in range(6):
        value = tick/5
        x1,y1=xy(value,0); x2,y2=xy(value,1)
        out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="black" stroke-opacity="0.15"/>'.format(x1,y1,x2,y2))
        x3,y3=xy(0,value); x4,y4=xy(1,value)
        out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="black" stroke-opacity="0.15"/>'.format(x3,y3,x4,y4))
        out.append('<text x="{:.1f}" y="{:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{:.1f}</text>'.format(x1,top+plot_h+24,value))
        out.append('<text x="{}" y="{:.1f}" text-anchor="end" font-family="Arial" font-size="12">{:.1f}</text>'.format(left-10,y3+4,value))
    x1,y1=xy(0,0); x2,y2=xy(1,1)
    out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="black" stroke-dasharray="6 5"/>'.format(x1,y1,x2,y2))
    for method, dash, shape in [('raw','', 'circle'),('calibrated','7 4','square')]:
        pts=[]
        for r in reliability_rows:
            if r['regime']==regime and r['method']==method:
                pts.append((float(r['mean_confidence']),float(r['observed_accuracy']),int(r['n'])))
        pts.sort()
        if pts:
            coords=[xy(x,y) for x,y,_ in pts]
            out.append('<polyline points="{}" fill="none" stroke="black" stroke-width="2"{} />'.format(' '.join('{:.1f},{:.1f}'.format(x,y) for x,y in coords), ' stroke-dasharray="{}"'.format(dash) if dash else ''))
            for (cx,cy,n),(x,y) in zip(pts,coords):
                radius=3+min(4,math.sqrt(n)/20)
                if shape=='circle': out.append('<circle cx="{:.1f}" cy="{:.1f}" r="{:.1f}" fill="white" stroke="black" stroke-width="2"/>'.format(x,y,radius))
                else: out.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" fill="white" stroke="black" stroke-width="2"/>'.format(x-radius,y-radius,2*radius,2*radius))
    out.append('<text x="{}" y="{}" font-family="Arial" font-size="13">Mean predicted correctness probability</text>'.format(left+60,height-35))
    out.append('<text x="22" y="{}" transform="rotate(-90 22 {})" text-anchor="middle" font-family="Arial" font-size="13">Observed correctness rate</text>'.format(top+plot_h/2,top+plot_h/2))
    out.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-dasharray="6 5"/>'.format(left+5, height-65,left+35,height-65))
    out.append('<text x="{}" y="{}" font-family="Arial" font-size="12">Perfect calibration</text>'.format(left+42,height-61))
    out.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black"/>'.format(left+220,height-65,left+250,height-65))
    out.append('<text x="{}" y="{}" font-family="Arial" font-size="12">Raw / calibrated curves use circles / squares</text>'.format(left+257,height-61))
    out.append('</svg>')
    Path(filename).write_text('\n'.join(out), encoding='utf-8')


def summarize_seeds(seed_summary):
    grouped = defaultdict(list)
    for row in seed_summary:
        grouped[(row["regime"], row["policy"])].append(row)
    summary = []
    for (regime, policy), values in sorted(grouped.items()):
        result = {"regime": regime, "policy": policy, "n": sum(int(x["n"]) for x in values)}
        for metric in ("accuracy", "brier", "ece"):
            numbers = [float(x[metric]) for x in values]
            result[metric] = statistics.mean(numbers)
            result[metric + "_sd_across_seeds"] = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
        summary.append(result)
    return summary


def write_csv(path, rows, fields):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(summary, metric, title, filename, y_max):
    """Create a monochrome SVG chart using only the standard library."""
    import html
    width, height, left, right, top, bottom = 1380, 560, 100, 60, 70, 145
    plot_w, plot_h = width - left - right, height - top - bottom
    regimes, policies = ["calibrated", "overconfident"], list(POLICIES)
    lookup = {(r["regime"], r["policy"]): float(r[metric]) for r in summary}
    xs = [left + plot_w * 0.28, left + plot_w * 0.72]
    styles = [("", "circle"), ("10 5", "square"), ("3 4", "triangle"), ("8 3 2 3", "diamond"), ("2 3", "cross")]
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(width, height, width, height),
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="{}" y="34" font-family="Arial" font-size="23" font-weight="bold">{}</text>'.format(left, html.escape(title))]
    for tick in range(6):
        value = y_max * tick / 5.0
        y = top + plot_h * (1.0 - value / y_max)
        out.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" stroke="black" stroke-opacity="0.18"/>'.format(left, y, width-right, y))
        out.append('<text x="{}" y="{:.1f}" text-anchor="end" font-family="Arial" font-size="13">{:.2f}</text>'.format(left-12, y+4, value))
    out += ['<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black"/>'.format(left, top, left, top+plot_h),
            '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black"/>'.format(left, top+plot_h, width-right, top+plot_h)]
    for regime, x in zip(regimes, xs):
        out.append('<text x="{}" y="{}" text-anchor="middle" font-family="Arial" font-size="15">{}</text>'.format(x, top+plot_h+28, regime.title()))
    out.append('<text x="24" y="{}" transform="rotate(-90 24 {})" text-anchor="middle" font-family="Arial" font-size="14">{}</text>'.format(top+plot_h/2, top+plot_h/2, html.escape(metric.replace('_', ' ').title())))
    for i, policy in enumerate(policies):
        points = []
        for regime, x in zip(regimes, xs):
            y = top + plot_h * (1.0 - lookup[(regime, policy)] / y_max)
            points.append((x, y))
        dash, marker = styles[i]
        dash_attr = ' stroke-dasharray="{}"'.format(dash) if dash else ''
        out.append('<polyline points="{}" fill="none" stroke="black" stroke-width="2.4"{} />'.format(' '.join('{:.1f},{:.1f}'.format(x,y) for x,y in points), dash_attr))
        for x, y in points:
            if marker == 'circle':
                out.append('<circle cx="{:.1f}" cy="{:.1f}" r="5" fill="white" stroke="black" stroke-width="2"/>'.format(x,y))
            elif marker == 'square':
                out.append('<rect x="{:.1f}" y="{:.1f}" width="9" height="9" fill="white" stroke="black" stroke-width="2"/>'.format(x-4.5,y-4.5))
            elif marker == 'triangle':
                out.append('<polygon points="{:.1f},{:.1f} {:.1f},{:.1f} {:.1f},{:.1f}" fill="white" stroke="black" stroke-width="2"/>'.format(x,y-6,x-6,y+5,x+6,y+5))
            elif marker == 'diamond':
                out.append('<polygon points="{:.1f},{:.1f} {:.1f},{:.1f} {:.1f},{:.1f} {:.1f},{:.1f}" fill="white" stroke="black" stroke-width="2"/>'.format(x,y-6,x+6,y,x,y+6,x-6,y))
            else:
                out.append('<path d="M {:.1f} {:.1f} l 10 0 M {:.1f} {:.1f} l 0 10" stroke="black" stroke-width="2"/>'.format(x-5,y,x,y-5))
    legend_y = height - 78
    for i, policy in enumerate(policies):
        x = left + i * 250
        dash, marker = styles[i]
        dash_attr = ' stroke-dasharray="{}"'.format(dash) if dash else ''
        out.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="2.4"{} />'.format(x,legend_y,x+36,legend_y,dash_attr))
        if marker == 'circle':
            out.append('<circle cx="{}" cy="{}" r="4" fill="white" stroke="black" stroke-width="2"/>'.format(x+18,legend_y))
        elif marker == 'square':
            out.append('<rect x="{}" y="{}" width="8" height="8" fill="white" stroke="black" stroke-width="2"/>'.format(x+14,legend_y-4))
        elif marker == 'triangle':
            out.append('<polygon points="{},{} {},{} {},{}" fill="white" stroke="black" stroke-width="2"/>'.format(x+18,legend_y-5,x+13,legend_y+4,x+23,legend_y+4))
        elif marker == 'diamond':
            out.append('<polygon points="{},{} {},{} {},{} {},{}" fill="white" stroke="black" stroke-width="2"/>'.format(x+18,legend_y-5,x+23,legend_y,x+18,legend_y+5,x+13,legend_y))
        else:
            out.append('<path d="M {} {} l 10 0 M {} {} l 0 10" stroke="black" stroke-width="2"/>'.format(x+13,legend_y,x+18,legend_y-5))
        out.append('<text x="{}" y="{}" font-family="Arial" font-size="10">{}</text>'.format(x+44,legend_y+4,html.escape(policy.replace('_',' '))))
    out.append('<text x="{}" y="{}" font-family="Arial" font-size="12">Lower is better for Brier/ECE; higher is better for accuracy.</text>'.format(left,height-25))
    out.append('</svg>')
    with open(filename, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(out))
