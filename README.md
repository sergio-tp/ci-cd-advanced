# revolutionary-mlops

A revolutionary ML project that trains a model guaranteed to return `TRUE` for any input. Powered by cutting-edge Gaussian statistics and the world's most optimistic threshold classifier.

## Requirements

- [uv](https://github.com/astral-sh/uv)

## Setup

```bash
uv run --python 3.14 pytest -q
```

## Running

### Train

```bash
uv run -m revolutionary_mlops train [--train-path data/train.csv] [--test-path data/test.csv]
```

Trains the model, evaluates on test data and auto-magically stores it in a secure place. Prints the `model_id` to use for validation.

### Validate

```bash
uv run -m revolutionary_mlops validate <model_id> [--validate-path data/validate.csv]
```

Retrieves the model by `model_id` and evaluates on validation data, printing accuracy/precision/recall.

## CI/CD local equivalent

The CI pipeline runs all checks with `uv run`:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check .
uv run pytest -q
uv run python scripts/run_ci_pipeline.py --minimum-score 0.80
```

The last command trains a new model, validates it, enforces the 0.80 quality gate for accuracy/precision/recall, and writes:

- `artifacts/latest_metrics.json`
- `public/index.html` (GitHub Pages report)

### Data format

CSV files have no header. Each row starts with the target (`TRUE`) followed by a variable number of random features:

```
TRUE,0.4023,0.7235,0.3185
TRUE,0.1969,0.5479,0.6219,0.9172,0.2438,0.1050
```
