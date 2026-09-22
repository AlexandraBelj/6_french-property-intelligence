"""
PURPOSE
-------
Load the frozen French Property Intelligence production model.

Two controlled loading modes are supported:

1. Local development
   Uses the already validated local outputs/models/model.pkl.

2. Production deployment
   Downloads the canonical MLflow artifact from AWS S3.

In both cases the model file is verified against the frozen SHA-256
checksum BEFORE deserialization. This guarantees that the API serves
the same binary artifact that was validated and recorded in MLflow.

Only trusted project-owned pickle/joblib artifacts must ever be loaded.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import boto3
import joblib


# ---------------------------------------------------------------------
# Frozen production artifact identity
# ---------------------------------------------------------------------

EXPECTED_MODEL_SHA256 = (
    "eef836a91e493c06558f594ee4fd1a96110e88de6fc2eb9e23f0fd3cc843fe16"
)

DEFAULT_LOCAL_MODEL_PATH = Path("outputs/models/model.pkl")

PRODUCTION_MODEL_PATH = Path("/tmp/model.pkl")


class ModelLoadingError(RuntimeError):
    """Raised when the production model cannot be loaded safely."""


def _calculate_sha256(path: Path) -> str:
    """
    Calculate a file SHA-256 without loading the whole model into memory.
    """

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _verify_model_artifact(path: Path) -> None:
    """
    Verify that a model file exists and matches the frozen production hash.
    """

    if not path.exists():
        raise ModelLoadingError(
            f"Model artifact does not exist: {path}"
        )

    actual_sha256 = _calculate_sha256(path)

    if actual_sha256 != EXPECTED_MODEL_SHA256:
        raise ModelLoadingError(
            "Model artifact SHA-256 does not match the validated "
            "production artifact."
        )


def _download_model_from_s3(destination: Path) -> None:
    """
    Download the canonical production artifact from private AWS S3.

    Bucket, object key and AWS credentials are supplied through
    environment variables in the deployment environment.
    """

    bucket = os.getenv("S3_BUCKET")
    model_key = os.getenv("S3_MODEL_KEY")

    if not bucket:
        raise ModelLoadingError(
            "S3_BUCKET environment variable is not configured."
        )

    if not model_key:
        raise ModelLoadingError(
            "S3_MODEL_KEY environment variable is not configured."
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        s3 = boto3.client("s3")

        s3.download_file(
            bucket,
            model_key,
            str(destination),
        )

    except Exception as exc:
        raise ModelLoadingError(
            "Unable to download the production model from S3."
        ) from exc


def load_production_model() -> Any:
    """
    Load and return the validated production inference object.

    Environment
    -----------
    MODEL_SOURCE:
        "local" -> use outputs/models/model.pkl
        "s3"    -> download canonical model from AWS S3

    Defaults to "local" so development never performs an accidental
    337 MB S3 download.
    """

    model_source = os.getenv(
        "MODEL_SOURCE",
        "local",
    ).strip().lower()

    if model_source == "local":
        model_path = DEFAULT_LOCAL_MODEL_PATH

    elif model_source == "s3":
        model_path = PRODUCTION_MODEL_PATH

        # Avoid downloading the same large artifact again when the
        # container already has a verified local copy.
        if model_path.exists():
            try:
                _verify_model_artifact(model_path)
            except ModelLoadingError:
                model_path.unlink(missing_ok=True)
                _download_model_from_s3(model_path)
        else:
            _download_model_from_s3(model_path)

    else:
        raise ModelLoadingError(
            "MODEL_SOURCE must be either 'local' or 's3'."
        )

    # Integrity verification always happens before deserialization.
    _verify_model_artifact(model_path)

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise ModelLoadingError(
            "The validated model artifact could not be deserialized."
        ) from exc

    return model