"""
PURPOSE
-------
Register the frozen French Property Intelligence production artifact
and its final evaluation results in the deployed MLflow tracking server.

This script DOES NOT:
- retrain the models,
- tune hyperparameters,
- reopen model selection,
- access the held-out test dataset.

It logs:
1. Final model architecture and training configuration
2. Frozen validation and held-out test metrics
3. SHA-256 checksum of the production artifact
4. The validated production model.pkl itself

MLflow metadata is persisted in Neon PostgreSQL.
MLflow artifacts are persisted in AWS S3 through the MLflow server.
"""

from hashlib import sha256
from pathlib import Path

import mlflow


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TRACKING_URI = (
    "https://alexbelj-french-property-intelligence-mlflow.hf.space"
)

EXPERIMENT_NAME = "french-property-intelligence"
RUN_NAME = "final-production-model"

MODEL_PATH = Path("outputs/models/model.pkl")


# ---------------------------------------------------------------------
# Validate the local production artifact before uploading it.
# ---------------------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Production artifact not found: {MODEL_PATH.resolve()}"
    )

model_size_bytes = MODEL_PATH.stat().st_size

print(f"Production artifact: {MODEL_PATH.resolve()}")
print(f"Size: {model_size_bytes:,} bytes")


# ---------------------------------------------------------------------
# Calculate SHA-256.
#
# PURPOSE:
# The checksum gives us a stable identity for the exact binary artifact.
# Later we can verify that FastAPI is serving the same model that was
# recorded in MLflow.
#
# Read in chunks so the entire 337 MB file is not loaded into RAM.
# ---------------------------------------------------------------------

digest = sha256()

with MODEL_PATH.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
        digest.update(chunk)

model_sha256 = digest.hexdigest()

print(f"SHA-256: {model_sha256}")


# ---------------------------------------------------------------------
# Connect to the deployed MLflow tracking server.
# ---------------------------------------------------------------------

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ---------------------------------------------------------------------
# Log the frozen production model record.
# ---------------------------------------------------------------------

with mlflow.start_run(run_name=RUN_NAME) as run:

    # High-level model architecture and data design.
    mlflow.log_params(
        {
            "model_family": "LightGBM",
            "architecture": "specialized_house_apartment",
            "target": "transaction_price_eur",
            "primary_metric": "MAE",
            "split_strategy": "parcel_group_aware_random",
            "training_period_start": "2020-01-01",
            "training_period_end": "2024-06-30",
            "house_num_leaves": 127,
            "house_iterations": 5000,
            "apartment_num_leaves": 127,
            "apartment_iterations": 3752,
            "learning_rate": 0.05,
        }
    )

    # Frozen validation metrics used during final model selection.
    mlflow.log_metrics(
        {
            "house_validation_mae": 68636.3686,
            "house_validation_rmse": 137724.1749,
            "house_validation_r2": 0.6982323,
            "apartment_validation_mae": 47489.1,
        }
    )

    # Final held-out test metrics.
    #
    # These were measured only after model selection was frozen.
    mlflow.log_metrics(
        {
            "house_test_mae": 68296.0,
            "house_test_rmse": 136391.0,
            "house_test_r2": 0.7015,
            "apartment_test_mae": 47499.0,
            "apartment_test_rmse": 104031.0,
            "apartment_test_r2": 0.8239,
        }
    )

    # Artifact identity information.
    mlflow.set_tags(
        {
            "stage": "production_candidate",
            "artifact_filename": MODEL_PATH.name,
            "artifact_sha256": model_sha256,
            "artifact_size_bytes": str(model_size_bytes),
            "test_set_status": "closed_after_final_evaluation",
            "project_scope": "metropolitan_france_residential",
        }
    )

    print()
    print("Uploading production model to MLflow / S3...")
    print("This is a large artifact (~337 MB); allow the upload to finish.")

    # MLflow proxies this upload to the configured AWS S3 artifact store.
    mlflow.log_artifact(
        str(MODEL_PATH),
        artifact_path="production",
    )

    print()
    print("Production model logged successfully.")
    print(f"Run ID: {run.info.run_id}")
    print(f"Artifact SHA-256: {model_sha256}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")