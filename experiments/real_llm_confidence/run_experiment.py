#!/usr/bin/env python3
"""Real local LLM confidence: three agents answer yes/no questions about clinic notes.

Agents (all local through Ollama, temperature 0):
  qwen_plain    qwen2.5:3b with a plain prompt
  llama_plain   llama3.1:8b with the same plain prompt
  qwen_checker  qwen2.5:3b with a "careful checker" prompt (same model, different role)

Each agent answers every question in English and Roman Urdu with a confidence (50-100).
The script then measures each agent's calibration, how often agents are wrong together,
and the five team rules from src/uncertainty_benchmark.py with per-agent calibration
cross-fitted by note (the calibration never sees the notes it is scored on).

Usage (Python 3.9+, standard library only; Ollama must be running):
  python3 run_experiment.py            # call the models (resumes from outputs/replies.jsonl)
  python3 run_experiment.py --analyse  # only recompute the CSVs from saved replies
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.uncertainty_benchmark import POLICIES, _team_probability, expected_calibration_error, fit_logistic_calibrators  # noqa: E402

OUT = HERE / "outputs"
REPLIES = OUT / "replies.jsonl"
OLLAMA = "http://localhost:11434"
LANGS = ("en", "ru")
FOLDS = 4

PLAIN = ("You are an assistant in a physiotherapy clinic. Read the clinic note and answer the question "
         "using only the note. If the note does not say it, the answer is no. "
         'Reply with JSON only: {"answer": "yes" or "no", "confidence": a whole number from 50 to 100 '
         "showing how sure you are that your answer is right}.")
CHECKER = ("You are a careful checker of clinic notes. Before answering, check that the exact detail in the "
           "question (side of the body, number, time, treatment) is written in the note. Answer using only "
           "the note. If the note does not say it, the answer is no. "
           'Reply with JSON only: {"answer": "yes" or "no", "confidence": a whole number from 50 to 100 '
           "showing how sure you are that your answer is right}.")
AGENTS = {
    "qwen_plain": ("qwen2.5:3b", PLAIN),
    "llama_plain": ("llama3.1:8b", PLAIN),
    "qwen_checker": ("qwen2.5:3b", CHECKER),
}


# ------------------------------------------------------------------ model calls

def load_items() -> List[dict]:
    return json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))["items"]


def user_message(item: dict, lang: str) -> str:
    return "Clinic note:\n{}\n\nQuestion: {}".format(item[lang]["note"], item[lang]["question"])


def parse_reply(text: str) -> Tuple[Optional[int], Optional[float]]:
    """Return (predicted label 1=yes/0=no, confidence 0.5-1.0) or (None, None) if unusable."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    answer = str(data.get("answer", "")).strip().lower()
    if answer not in ("yes", "no"):
        return None, None
    try:
        conf = float(data.get("confidence"))
    except (TypeError, ValueError):
        return None, None
    if conf <= 1.0:  # some models answer 0.9 instead of 90
        conf *= 100.0
    conf = min(100.0, max(50.0, conf))
    return (1 if answer == "yes" else 0), conf / 100.0


def ollama_chat(model: str, system: str, user: str) -> str:
    body = json.dumps({"model": model, "stream": False, "format": "json",
                       "options": {"temperature": 0, "seed": 7},
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode("utf-8")
    req = urllib.request.Request(OLLAMA + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))["message"]["content"]


def collect(call: Callable[[str, str, str], str] = ollama_chat) -> None:
    OUT.mkdir(exist_ok=True)
    done = set()
    if REPLIES.exists():
        for line in REPLIES.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["agent"], r["item_id"], r["lang"]))
    items = load_items()
    todo = [(a, it, lang) for a in AGENTS for it in items for lang in LANGS if (a, it["item_id"], lang) not in done]
    print("{} calls to make ({} already saved)".format(len(todo), len(done)))
    with REPLIES.open("a", encoding="utf-8") as f:
        for n, (agent, item, lang) in enumerate(todo, 1):
            model, system = AGENTS[agent]
            start = time.time()
            text = call(model, system, user_message(item, lang))
            pred, conf = parse_reply(text)
            if pred is None:  # one retry with a reminder
                text = call(model, system, user_message(item, lang) + '\n\nReply with JSON only, for example {"answer": "no", "confidence": 80}.')
                pred, conf = parse_reply(text)
            f.write(json.dumps({"agent": agent, "model": model, "item_id": item["item_id"], "lang": lang,
                                "reply": text, "predicted": pred, "confidence": conf,
                                "seconds": round(time.time() - start, 2)}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 24 == 0 or n == len(todo):
                print("  {}/{}".format(n, len(todo)))


# ------------------------------------------------------------------ analysis

def load_rows() -> List[dict]:
    items = {it["item_id"]: it for it in load_items()}
    rows = []
    for line in REPLIES.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        it = items[r["item_id"]]
        truth = 1 if it["answer"] == "yes" else 0
        parsed = r["predicted"] is not None
        # An unusable reply counts as a wrong answer given with minimum confidence.
        pred = r["predicted"] if parsed else 1 - truth
        rows.append({"agent_id": r["agent"], "item_id": r["item_id"], "note_id": it["note_id"], "type": it["type"],
                     "lang": r["lang"], "truth": truth, "predicted_label": pred,
                     "reported_confidence": r["confidence"] if parsed else 0.5,
                     "actual_correct": int(pred == truth), "parsed": int(parsed)})
    return rows


def fold_of(note_id: str) -> int:
    return (int(note_id[1:]) - 1) % FOLDS


def agent_metrics(rows: List[dict]) -> List[dict]:
    out = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["agent_id"], r["lang"])].append(r)
        groups[(r["agent_id"], "all")].append(r)
    for (agent, lang), g in sorted(groups.items()):
        conf = [r["reported_confidence"] for r in g]
        corr = [r["actual_correct"] for r in g]
        out.append({"agent": agent, "lang": lang, "n": len(g),
                    "accuracy": round(statistics.mean(corr), 4),
                    "mean_confidence": round(statistics.mean(conf), 4),
                    "overconfidence": round(statistics.mean(conf) - statistics.mean(corr), 4),
                    "brier": round(statistics.mean((c - y) ** 2 for c, y in zip(conf, corr)), 4),
                    "ece": round(expected_calibration_error(conf, corr), 4),
                    "unusable_replies": sum(1 - r["parsed"] for r in g)})
    return out


def accuracy_by_type(rows: List[dict]) -> List[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["agent_id"], r["lang"], r["type"])].append(r["actual_correct"])
    return [{"agent": a, "lang": l, "type": t, "n": len(v), "accuracy": round(statistics.mean(v), 4)}
            for (a, l, t), v in sorted(groups.items())]


def error_overlap(rows: List[dict]) -> List[dict]:
    """How often two agents are wrong on the same question, against what independence would predict."""
    wrong = {(r["agent_id"], r["item_id"], r["lang"]): 1 - r["actual_correct"] for r in rows}
    keys = sorted({(r["item_id"], r["lang"]) for r in rows})
    out = []
    for a, b in combinations(AGENTS, 2):
        for lang in LANGS + ("all",):
            ks = [k for k in keys if lang == "all" or k[1] == lang]
            wa = [wrong[(a,) + k] for k in ks]
            wb = [wrong[(b,) + k] for k in ks]
            pa, pb = statistics.mean(wa), statistics.mean(wb)
            both = statistics.mean(x * y for x, y in zip(wa, wb))
            out.append({"pair": "{} + {}".format(a, b), "lang": lang, "n": len(ks),
                        "wrong_a": round(pa, 4), "wrong_b": round(pb, 4),
                        "both_wrong": round(both, 4), "both_wrong_if_independent": round(pa * pb, 4),
                        "ratio": round(both / (pa * pb), 2) if pa * pb > 0 else ""})
    return out


def team_predictions(rows: List[dict]) -> List[dict]:
    """Five team rules; per-agent calibration fitted on the other folds' notes only."""
    preds = []
    for fold in range(FOLDS):
        cal_rows = []
        for r in rows:
            cal_rows.append(dict(r, seed=fold, regime="real",
                                 split="test" if fold_of(r["note_id"]) == fold else "calibration"))
        calibrators = fit_logistic_calibrators(cal_rows)
        by_q = defaultdict(list)
        for r in cal_rows:
            if r["split"] == "test":
                by_q[(r["item_id"], r["lang"])].append(r)
        for (item_id, lang), agent_rows in sorted(by_q.items()):
            truth = agent_rows[0]["truth"]
            for policy in POLICIES:
                p = _team_probability(agent_rows, policy, calibrators)
                preds.append({"fold": fold, "item_id": item_id, "lang": lang, "policy": policy,
                              "truth": truth, "p_yes": round(p, 6)})
    return preds


def team_metrics(preds: List[dict]) -> List[dict]:
    out = []
    for policy in POLICIES:
        for lang in LANGS + ("all",):
            g = [p for p in preds if p["policy"] == policy and (lang == "all" or p["lang"] == lang)]
            label = [1 if p["p_yes"] > 0.5 else 0 for p in g]  # a 50/50 tie counts as "no"
            conf = [max(p["p_yes"], 1 - p["p_yes"]) for p in g]
            corr = [int(l == p["truth"]) for l, p in zip(label, g)]
            out.append({"policy": policy, "lang": lang, "n": len(g),
                        "accuracy": round(statistics.mean(corr), 4),
                        "brier": round(statistics.mean((p["p_yes"] - p["truth"]) ** 2 for p in g), 4),
                        "ece": round(expected_calibration_error(conf, corr), 4)})
    return out


def write_csv(name: str, rows: List[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def ollama_version() -> str:
    try:
        with urllib.request.urlopen(OLLAMA + "/api/version", timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8")).get("version", "unknown")
    except OSError:
        return ""


def analyse() -> None:
    rows = load_rows()
    expected = len(AGENTS) * len(load_items()) * len(LANGS)
    if len(rows) != expected:
        raise SystemExit("Expected {} replies, found {}. Run without --analyse first.".format(expected, len(rows)))
    am = agent_metrics(rows)
    write_csv("agent_metrics.csv", am)
    write_csv("accuracy_by_type.csv", accuracy_by_type(rows))
    write_csv("error_overlap.csv", error_overlap(rows))
    preds = team_predictions(rows)
    write_csv("team_predictions.csv", preds)
    tm = team_metrics(preds)
    write_csv("team_metrics.csv", tm)
    print("\nAgent results (all questions):")
    for r in am:
        if r["lang"] == "all":
            print("  {agent:13s} accuracy {accuracy:.3f}  mean confidence {mean_confidence:.3f}  "
                  "ECE {ece:.3f}  unusable {unusable_replies}".format(**r))
    print("\nTeam results (all questions):")
    for r in tm:
        if r["lang"] == "all":
            print("  {policy:32s} accuracy {accuracy:.3f}  Brier {brier:.3f}  ECE {ece:.3f}".format(**r))
    print("\nCSV files written to {}".format(OUT))


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--analyse", action="store_true", help="skip model calls; analyse saved replies")
    args = ap.parse_args(argv)
    if not args.analyse:
        version = ollama_version()
        if not version:
            print("Ollama is not running. Open the Ollama app (or run 'ollama serve') and try again.")
            return 1
        collect()
        (OUT / "RUN_MANIFEST.json").write_text(json.dumps({
            "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "ollama_version": version, "agents": {a: m for a, (m, _) in AGENTS.items()},
            "temperature": 0, "seed": 7, "items": len(load_items()), "languages": list(LANGS),
            "folds_by_note": FOLDS}, indent=2) + "\n", encoding="utf-8")
    analyse()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
