# Real local LLM confidence in English and Roman Urdu clinic notes

The earlier experiments in this repository use simulated agents. This one asks the same questions with real local models.

## Research questions

1. When small local LLMs say how sure they are, does that confidence match how often they are right, in English and in Roman Urdu?
2. Are two agents built on the same model wrong together more often than two different models?
3. Which of the five team rules from the synthetic benchmark still works with real model answers?

## Agents

All run locally with Ollama at temperature 0.

| Agent | Model | Prompt |
|---|---|---|
| `qwen_plain` | qwen2.5:3b | plain clinic assistant |
| `llama_plain` | llama3.1:8b | plain clinic assistant |
| `qwen_checker` | qwen2.5:3b | careful checker (check side, number, time, treatment) |

`qwen_plain` and `qwen_checker` share a model on purpose: the pair tests whether a different role is enough to make errors independent.

## Tasks

`tasks.json` has 12 invented physiotherapy notes (not real patients), each written in English and Roman Urdu, with 4 yes/no questions per note:

| Type | Example | Answer |
|---|---|---|
| `stated` | Is the pain worse when sitting? | yes |
| `negation` | Does the note say there is no numbness in the legs? | yes |
| `wrong_detail` | Has the back pain lasted 3 months? (note says 3 weeks) | no |
| `not_in_note` | Has the patient had an X-ray? (not mentioned) | no |

48 questions x 2 languages x 3 agents = 288 model calls. `not_in_note` questions test the same failure as unsupported facts in clinical notes: saying yes to something the note never says.

## Run (Mac, Python 3.9+, standard library only)

```bash
ollama pull qwen2.5:3b && ollama pull llama3.1:8b   # once
python3 run_experiment.py                           # about 10 to 15 minutes on a MacBook M4
python3 -m unittest discover -s tests -v
```

The run saves every reply to `outputs/replies.jsonl` and can be stopped and restarted; it continues where it stopped. `python3 run_experiment.py --analyse` recomputes the tables from saved replies.

## Method

- Confidence is the number the model writes (50 to 100). A reply that is not valid JSON gets one retry; if it is still unusable it is scored as a wrong answer with confidence 0.5 and counted in `unusable_replies`.
- Calibration (ECE, 10 bins) and Brier score as in the synthetic benchmark.
- Error overlap: how often two agents are wrong on the same question, against the rate expected if their errors were independent (`ratio` above 1 means they fail together).
- Team rules: the five policies from `src/uncertainty_benchmark.py`. Per-agent calibration is cross-fitted in 4 folds by note, so it is never fitted on the note it is scored on.

## Outputs

`agent_metrics.csv`, `accuracy_by_type.csv`, `error_overlap.csv`, `team_metrics.csv`, `team_predictions.csv`, `replies.jsonl`, `RUN_MANIFEST.json`.

## Results

Run on 24 October 2026 on a MacBook M4 (16 GB) with Ollama 0.40.10; 288 replies, none unusable. Numbers from `outputs/`.

| Agent | Language | Accuracy | Mean confidence | ECE |
|---|---|---|---|---|
| qwen_plain | English | 95.8% | 99.2% | 0.050 |
| qwen_plain | Roman Urdu | 85.4% | 99.4% | 0.140 |
| llama_plain | English | 97.9% | 100% | 0.021 |
| llama_plain | Roman Urdu | 91.7% | 100% | 0.083 |
| qwen_checker | English | 97.9% | 98.5% | 0.027 |
| qwen_checker | Roman Urdu | 87.5% | 99.0% | 0.115 |

- **Confidence carried almost no information:** 20 of the 21 wrong answers had confidence 100 (the other had 90). llama3.1:8b gave 100 on all 96 answers.
- **Left and right failed in Roman Urdu only:** on the 4 questions that name the wrong side of the body, all three agents answered "yes" on 3 (knee, heel, arm) and both qwen agents on the fourth (shoulder). In English all 4 were answered correctly by every agent.
- **No "yes" to things not in the note:** all 24 `not_in_note` questions were answered correctly by every agent.
- **Shared errors:** in Roman Urdu, `qwen_plain` and `qwen_checker` were both wrong on 12.5% of questions, against 1.8% expected if independent (ratio 6.9). Every `qwen_checker` error was also a `qwen_plain` error. The cross-model pairs had ratios of 5.1 and 6.0.
- **Team rules:** all five rules scored 100% in English and 87.5% in Roman Urdu, below `llama_plain` alone (91.7%). With confidence almost always 100, calibration had nothing to work with.

## Observations

1. A wrong side in a clinical note is a serious safety risk: a plan for the wrong knee could harm a patient. Here the side errors appeared only in Roman Urdu, which matters for MeetPhysio. These were invented notes and a yes/no test, not output from MeetPhysio.
2. When a model gives 100% confidence on almost every answer, right or wrong, the score tells me almost nothing. I would not use verbal confidence from these models to decide when to trust an agent; a separate check is needed.
3. The checker used the same model, so a new role did not change its mistakes. My guess is that the weakness is in how the model reads Roman Urdu side words, possibly at tokenisation level, but this is not tested. A different model (llama3.1:8b) made fewer mistakes but still failed the same three side questions, so changing the model alone is not enough either.

Write-up: [Three local AI models mixed up left and right in Roman Urdu, and were sure they were right](https://masmatixinnovations.co.uk/insights/roman-urdu-left-right-confidence.html)

## Limitations

This is a pilot: 48 questions per language, written by one person, one run at temperature 0. Each question is about 2 percentage points of accuracy, so small differences are not meaningful. Verbal confidence is only one way to get uncertainty from a model; token probabilities are not used here. Roman Urdu spelling varies and only one spelling is tested.
