---
title: French Property Intelligence API
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# French Property Intelligence API

FastAPI deployment for the French Property Intelligence project.

## Endpoints

- `GET /health` — verifies that the API is running and the production model is loaded.
- `POST /predict` — geocodes a French address and returns an indicative residential property valuation.

## Production model

The API does not contain the model binary in Git or in the Docker image.

At startup, it downloads the frozen production `model.pkl` artifact from the private AWS S3 artifact store and verifies its SHA-256 checksum before deserialization.

The production artifact is the same artifact recorded by the project's MLflow experiment.

## Architecture

User request → FastAPI → address geocoding → frozen production inference pipeline → estimated price.

MLflow tracks the production model metadata and evaluation results.

Neon PostgreSQL stores MLflow metadata.

AWS S3 stores the production model artifact.

## Scope

The API provides indicative residential property valuation for houses and apartments in metropolitan France.

It is a decision-support tool and is not a professional or legal property appraisal.