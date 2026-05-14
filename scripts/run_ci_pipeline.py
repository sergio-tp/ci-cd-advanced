import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from revolutionary_mlops.pipelines.train_pipeline import run_training_pipeline
from revolutionary_mlops.pipelines.validate_pipeline import run_validation_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, validate and generate report.")
    parser.add_argument("--minimum-score", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_path = Path("data/train.csv")
    test_path = Path("data/test.csv")
    validate_path = Path("data/validate.csv")
    json_output = Path("artifacts/latest_metrics.json")
    html_output = Path("public/index.html")

    model_id = run_training_pipeline(
        train_path=train_path,
        test_path=test_path,
    )
    metrics = run_validation_pipeline(
        model_id=model_id,
        validate_path=validate_path,
    )

    passed = (
        metrics["accuracy"] >= args.minimum_score
        and metrics["precision"] >= args.minimum_score
        and metrics["recall"] >= args.minimum_score
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    model_threshold = float(model_id.split("_", maxsplit=1)[1])
    payload = {
        "generated_at": generated_at,
        "status": "passed" if passed else "failed",
        "minimum_score": args.minimum_score,
        "model_id": model_id,
        "model_threshold": model_threshold,
        "metrics": metrics,
    }

    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    status = "PASSED" if passed else "FAILED"
    html_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Validation report</title>
</head>
<body>
  <h1>Validation report</h1>
  <p><strong>Status:</strong> {status}</p>
  <p><strong>Generated at:</strong> {generated_at}</p>
  <p><strong>Model ID:</strong> {model_id}</p>
  <p><strong>Model threshold:</strong> {model_threshold:.4f}</p>
  <p><strong>Minimum accepted score:</strong> {args.minimum_score:.2f}</p>
  <table border="1" cellspacing="0" cellpadding="6">
    <tr><td>accuracy</td><td>{metrics["accuracy"]:.4f}</td></tr>
    <tr><td>precision</td><td>{metrics["precision"]:.4f}</td></tr>
    <tr><td>recall</td><td>{metrics["recall"]:.4f}</td></tr>
    <tr><td>total</td><td>{metrics["total"]}</td></tr>
    <tr><td>tp</td><td>{metrics["tp"]}</td></tr>
    <tr><td>fp</td><td>{metrics["fp"]}</td></tr>
    <tr><td>fn</td><td>{metrics["fn"]}</td></tr>
    <tr><td>tn</td><td>{metrics["tn"]}</td></tr>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )

    print(f"report_json={json_output}")
    print(f"report_html={html_output}")

    if not passed:
        raise SystemExit(
            "Quality gate failed: accuracy, precision and recall must all be >= "
            f"{args.minimum_score:.2f}"
        )


if __name__ == "__main__":
    main()
