"""Run from experiments/real_llm_confidence: python3 -m unittest discover -s tests -v"""
from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiment as rx  # noqa: E402


class ParseTest(unittest.TestCase):

    def test_good_and_bad_replies(self) -> None:
        self.assertEqual(rx.parse_reply('{"answer": "Yes", "confidence": 90}'), (1, 0.9))
        self.assertEqual(rx.parse_reply('{"answer": "no", "confidence": 0.8}'), (0, 0.8))
        self.assertEqual(rx.parse_reply('{"answer": "no", "confidence": 20}'), (0, 0.5))   # clipped to 50
        self.assertEqual(rx.parse_reply('{"answer": "no", "confidence": 140}'), (0, 1.0))
        for bad in ["", "yes", "[1]", '{"answer": "maybe", "confidence": 70}', '{"answer": "yes"}']:
            self.assertEqual(rx.parse_reply(bad), (None, None), bad)


class TaskFileTest(unittest.TestCase):

    def test_tasks_are_balanced_and_complete(self) -> None:
        items = rx.load_items()
        self.assertEqual(len({i["item_id"] for i in items}), len(items))
        self.assertEqual(Counter(i["answer"] for i in items), Counter({"yes": 24, "no": 24}))
        self.assertEqual(Counter(i["type"] for i in items)["not_in_note"], 12)
        for i in items:
            for lang in rx.LANGS:
                self.assertTrue(i[lang]["note"] and i[lang]["question"].endswith("?"))
        self.assertEqual({rx.fold_of(i["note_id"]) for i in items}, set(range(rx.FOLDS)))


class PipelineTest(unittest.TestCase):

    def test_fake_models_end_to_end(self) -> None:
        answers = {i["item_id"]: i["answer"] for i in rx.load_items()}
        rng = random.Random(1)

        def fake_call(model: str, system: str, user: str) -> str:
            # Look up the right answer from the question text, then be wrong 25% of the time.
            item = next(i for i in rx.load_items() if i["en"]["question"] in user or i["ru"]["question"] in user)
            ans = answers[item["item_id"]]
            if rng.random() < 0.25:
                ans = "no" if ans == "yes" else "yes"
            if rng.random() < 0.02:
                return "not json"
            return json.dumps({"answer": ans, "confidence": rng.choice([70, 80, 90, 95])})

        with tempfile.TemporaryDirectory() as d:
            rx.OUT = Path(d)
            rx.REPLIES = rx.OUT / "replies.jsonl"
            rx.collect(fake_call)
            rx.collect(fake_call)  # resume: nothing left to do
            lines = rx.REPLIES.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), len(rx.AGENTS) * 48 * 2)
            rx.analyse()
            team = (rx.OUT / "team_metrics.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(team), 1 + 5 * 3)
            overlap = (rx.OUT / "error_overlap.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(overlap), 1 + 3 * 3)
            preds = (rx.OUT / "team_predictions.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(preds), 1 + 96 * 5)  # every question scored once, out of fold


if __name__ == "__main__":
    unittest.main()
