"""
PURPOSE
-------
Validate the deployed MLflow infrastructure end-to-end before logging
the production property valuation model.

This test verifies:
1. Local client -> Hugging Face MLflow tracking server
2. MLflow run metadata -> Neon PostgreSQL
3. MLflow artifacts -> AWS S3

No production model and no sensitive data are used.
"""

import tempfile
from pathlib import Path

import mlflow


TRACKING_URI = (
    "https://alexbelj-french-property-intelligence-mlflow.hf.space"
)

EXPERIMENT_NAME = "infrastructure-validation"


# Point the local MLflow client at our deployed tracking server.
mlflow.set_tracking_uri(TRACKING_URI)

# Create/reuse a dedicated infrastructure validation experiment.
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="neon-s3-connectivity-test") as run:

    # Small metadata payload -> should be persisted by MLflow in Neon.
    mlflow.log_param("purpose", "infrastructure_validation")
    mlflow.log_metric("connectivity_test", 1.0)

    # Tiny harmless file -> should ultimately be stored in AWS S3.
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = Path(tmp_dir) / "connection_test.txt"
        artifact_path.write_text(
            "French Property Intelligence MLflow artifact test.\n",
            encoding="utf-8",
        )

        mlflow.log_artifact(
            str(artifact_path),
            artifact_path="validation",
        )

    print("MLflow infrastructure test succeeded.")
    print(f"Run ID: {run.info.run_id}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")