---
title: French Property Intelligence MLflow
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# French Property Intelligence — MLflow

MLflow tracking server for the French Property Intelligence certification project.

## Architecture

- **Tracking server:** MLflow on Hugging Face Spaces
- **Backend store:** Neon PostgreSQL
- **Artifact store:** AWS S3
- **Container:** Docker

Secrets and database credentials are configured through Hugging Face Spaces
environment secrets and are never stored in this repository.