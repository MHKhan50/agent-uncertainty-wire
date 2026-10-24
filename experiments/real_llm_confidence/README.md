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

Not run yet. This section will be filled from `outputs/` after the first run.

| Agent | Language | Accuracy | Mean confidence | ECE |
|---|---|---|---|---|
| qwen_plain | English | _pending_ | _pending_ | _pending_ |
| qwen_plain | Roman Urdu | _pending_ | _pending_ | _pending_ |
| llama_plain | English | _pending_ | _pending_ | _pending_ |
| llama_plain | Roman Urdu | _pending_ | _pending_ | _pending_ |
| qwen_checker | English | _pending_ | _pending_ | _pending_ |
| qwen_checker | Roman Urdu | _pending_ | _pending_ | _pending_ |

## Limitations

This is a pilot: 48 questions per language, written by one person, one run at temperature 0. Each question is about 2 percentage points of accuracy, so small differences are not meaningful. Verbal confidence is only one way to get uncertainty from a model; token probabilities are not used here. Roman Urdu spelling varies and only one spelling is tested.
