French Property Intelligence

Overview

French Property Intelligence is an end-to-end machine-learning application for residential property valuation in metropolitan France.

The project addresses a practical question:

Given a residential property's characteristics and location, what is a reasonable indicative sale value?

The solution combines data analysis, machine learning, experiment tracking, cloud storage, API development, geospatial services, generative AI, containerization, and cloud deployment.

Two specialized regression models estimate the value of houses and apartments. The production application then enriches the valuation with address resolution, nearby amenities, neighborhood exploration through Street View, and an AI-generated property description based only on verified structured information.

The application is designed as a decision-support tool. Its predictions are indicative statistical estimates and should not be interpreted as professional or legal property appraisals.

This project was developed as part of the Jedha Data Science certification.

Live Applications & Services

Component

Access

French Property Intelligence — Streamlit

https://huggingface.co/spaces/AlexBelj/french-property-intelligence

Valuation API

https://alexbelj-french-property-intelligence-api.hf.space

Interactive API Documentation

https://alexbelj-french-property-intelligence-api.hf.space/docs

MLflow Tracking Server

https://alexbelj-french-property-intelligence-mlflow.hf.space

MLflow Production Run

https://alexbelj-french-property-intelligence-mlflow.hf.space/#/experiments/2/runs/5b05b053d6384752bf565505fa222410

The Streamlit application, FastAPI service, and MLflow tracking server are deployed on Hugging Face Spaces.

MLflow uses Neon PostgreSQL as its backend metadata store and AWS S3 as its artifact store. The production model artifact is also stored in S3 and loaded by the FastAPI service at runtime.

Business Problem

Residential property valuation depends on both the characteristics of a property and its geographic context.

The objective of this project is to build a production-ready system able to estimate an indicative sale value from a compact set of information available to a user:

property type;

habitable surface;

number of rooms;

VEFA status;

address / geographic location.

The project deliberately separates two concerns:

Valuation — supervised machine learning estimates the property's value.

Property intelligence — external geospatial services and generative AI provide contextual information around that estimate.

This distinction is important:

The machine-learning models determine the estimated value. Generative AI does not participate in price prediction.

The GenAI layer only transforms verified structured property, valuation, and environment information into natural-language text.

Dataset

The modeling data is based on historical French residential property transactions.

The initial transaction dataset contains approximately:

9.14 million transactions;

transactions from 2014-01-01 to 2024-06-30;

houses and apartments;

transaction prices;

habitable surfaces;

room counts;

property types;

VEFA information;

geographic attributes.

After restricting the geographic scope to metropolitan France, approximately 9.05 million records remain.

For modeling, the project focuses on the more recent period:

2020-01-01 → 2024-06-30

After cleaning and modeling filters, the master modeling dataset contains approximately:

4.13 million transactions

The raw and processed datasets are intentionally not committed to the GitHub repository.

Data Preparation & Leakage Prevention

Property-transaction data requires careful cleaning because extreme or inconsistent observations can strongly affect regression models.

The modeling preparation includes:

restriction to houses and apartments;

metropolitan-France filtering;

recent transaction-period selection;

target and surface quality controls;

treatment of implausible room counts;

geographic feature preparation;

categorical normalization.

Training-quality filters include a price-per-square-meter range of:

€100/m² → €30,000/m²

and a habitable-surface range of:

10 m² → 500 m²

Extreme room-count anomalies are treated as missing rather than being interpreted as reliable property information.

Leakage Prevention

The project explicitly avoids using information that would reveal the target or create unrealistic evaluation conditions.

The modeling strategy includes:

no direct use of transaction price-derived predictors at inference time;

no valuation-date predictor in the final production model;

grouping by parcel when constructing train/validation/test partitions;

no parcel overlap between those partitions;

separate validation and final held-out test evaluation.

The project is treated as a cross-sectional property valuation problem, rather than a time-series forecasting problem.

Exploratory Data Analysis

The EDA focuses on questions that materially affect data preparation and modeling rather than producing exhaustive descriptive charts.

Important observations include:

houses represent approximately 55% of metropolitan transactions;

apartments represent approximately 45%;

VEFA transactions account for roughly 4%;

the price distribution is strongly right-skewed;

the metropolitan median transaction value is approximately €170,000.

Because high-value transactions create a long upper tail, Mean Absolute Error (MAE) is used as the primary evaluation metric.

RMSE and R² are retained as complementary metrics.

Machine-Learning Strategy

Houses and apartments have materially different market structures.

The project therefore uses two specialized production models rather than forcing both property types through a single regression model:

User property
      │
      ├── House ───────→ House LightGBM model
      │
      └── Apartment ───→ Apartment LightGBM model

Both final models use LightGBM regression with an L1 objective.

House Features

The house model uses:

habitable surface;

number of rooms;

VEFA status;

latitude;

longitude;

postcode;

virtual neighborhood information.

Apartment Features

The apartment model uses:

habitable surface;

number of rooms;

VEFA status;

latitude;

longitude;

postcode.

The virtual-neighborhood feature is used only for the house model.

Train / Validation / Test Design

The cleaned data is divided into separate training, validation, and held-out test partitions.

Houses

Training:    1,630,944
Validation:    349,176
Test:          349,001

Apartments

Training:    1,263,001
Validation:    269,977
Test:          268,358

The final test sets remain isolated from model tuning and are used only for final performance assessment.

Baselines

Simple baselines establish whether more advanced models provide meaningful predictive value.

Naive Baseline

Property type

MAE

House

€133,559

Apartment

€122,966

Ridge Regression

Property type

MAE

RMSE

R²

House

€124,039

€223,815

0.203

Apartment

€122,592

€212,375

0.266

These baselines provide a quantitative reference for evaluating the final LightGBM models.

Final Model Performance

The final models are evaluated on held-out test data.

Property type

Model

MAE

RMSE

R²

House

LightGBM

€68,296

€136,391

0.702

Apartment

LightGBM

€47,499

€104,031

0.824

Compared with the tested baselines, the LightGBM models substantially reduce absolute prediction error and explain a much larger share of observed price variation.

Validation Performance

The house model achieved a validation MAE of approximately:

€68,636

The apartment model achieved a validation MAE of approximately:

€47,489

The similarity between validation and held-out test performance provides an additional check on model generalization within the evaluated historical population.

Production Inference Pipeline

The production models are wrapped in a dedicated inference class:

PropertyValuationModel

The wrapper centralizes the transformations required for prediction and routes each property to the appropriate specialized model.

Its responsibilities include:

property-type normalization;

input validation;

postcode normalization;

categorical handling;

geographic transformation;

virtual-neighborhood assignment for houses;

house/apartment model routing;

final price prediction.

The serialized production artifact is:

model.pkl

The model binary is deliberately excluded from Git.

The production artifact is stored in AWS S3 and its integrity is checked before loading using a fixed SHA-256 hash.

This helps ensure that the deployed API loads the same validated model artifact that was produced during the modeling workflow.

Property Intelligence Features

The Streamlit application extends the statistical valuation with contextual information useful for exploring a property.

1. Property Valuation

The user enters:

address;

property type;

habitable surface;

number of rooms;

VEFA status.

The address is geocoded, the appropriate machine-learning model is selected, and an indicative estimated sale value is returned.

2. Property Environment

The application displays nearby amenities around the property, including:

public transport;

parks and green spaces;

supermarkets.

Nearby-place information is obtained from Geoapify using the geocoded property coordinates.

The interface displays the surrounding environment on an interactive map together with nearby-place names and distances.

3. Neighborhood Street View

A dedicated tab provides access to Google Street View so the user can visually explore the neighborhood around the resolved property location.

The available panorama may correspond to the nearest Street View coverage rather than the exact building entered by the user.

4. GenAI Property Description

The application can generate a short French property description using a Hugging Face-hosted language model.

The production generation model is:

Qwen/Qwen3.8-27B

The language model receives only structured information already available to the application, such as:

property type;

surface;

number of rooms;

VEFA status;

resolved address;

estimated value;

nearby verified points of interest.

The prompt explicitly prevents the model from inventing unsupported characteristics such as:

balcony or terrace;

floor;

parking or garage;

elevator;

view or exposure;

luminosity;

renovation status;

construction period;

DPE;

interior equipment.

The generated text is therefore a presentation layer, not an additional valuation model.

MLflow Experiment Tracking

MLflow is used to preserve the production-model experiment and associated metrics and artifacts.

The production MLflow server is deployed on Hugging Face Spaces:

https://alexbelj-french-property-intelligence-mlflow.hf.space

The experiment is:

french-property-intelligence

The final production run is:

final-production-model

MLflow records the production model information and evaluation metrics, while the model artifact is stored in AWS S3.

MLflow Architecture

                    ┌──────────────────────┐
                    │   MLflow Server      │
                    │ Hugging Face Space   │
                    └──────────┬───────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ Neon PostgreSQL  │      │      AWS S3      │
        │ backend metadata │      │ model/artifacts  │
        └──────────────────┘      └──────────────────┘

This separates experiment metadata from binary artifacts and keeps large model files outside the Git repository.

FastAPI Service

The production inference and context services are exposed through FastAPI.

Interactive documentation is available at:

https://alexbelj-french-property-intelligence-api.hf.space/docs

Main Endpoints

POST /predict
GET  /location
GET  /environment
POST /description

The responsibilities remain deliberately separated:

/predict performs property valuation;

/location resolves an address into geographic information;

/environment retrieves nearby points of interest;

/description generates natural-language text from structured facts.

Prediction Request

A valuation request contains the property characteristics required by the production model.

Example:

{
  "property_type": "apartment",
  "surface_habitable": 65,
  "n_pieces": 3,
  "vefa": false,
  "latitude": 48.862442,
  "longitude": 2.3362,
  "postcode": "75001"
}

The API returns the model's estimated property value.

FastAPI automatically exposes OpenAPI/Swagger documentation at /docs, where the endpoints can be inspected and tested interactively.

Streamlit Application

The public user interface is deployed at:

https://huggingface.co/spaces/AlexBelj/french-property-intelligence

The interface is organized around four complementary views:

Estimation
Environnement
Quartier
Description

A successful valuation is retained across the interface so the user can move between the different property-intelligence views without recomputing the prediction.

The Streamlit application does not load the machine-learning model directly.

Instead:

Streamlit → FastAPI → production model

This keeps the user interface separate from the inference service and provides a clear production API boundary.

Deployment Architecture

The complete production architecture is:

                         GitHub
                    source of truth
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     Streamlit          FastAPI           MLflow
     HF Space           HF Space          HF Space
          │                │                │
          │                │         ┌──────┴──────┐
          │                │         │             │
          │                │         ▼             ▼
          │                │       Neon          AWS S3
          │                │     PostgreSQL     artifacts
          │                │
          └───────────────►│
                           │
             ┌─────────────┼──────────────┐
             │             │              │
             ▼             ▼              ▼
         Geocoding     Geoapify      HF Inference
          service       Places          Qwen
                           │
                           ▼
                    production model
                       from S3

GitHub is the source of truth for application code.

Hugging Face Spaces contain deployment copies of the services. Secrets and cloud credentials are configured through deployment environment variables/secrets and are not stored in Git.

Technologies

Data & Machine Learning

Python

pandas

NumPy

SciPy

scikit-learn

LightGBM

PyProj

joblib

Jupyter Notebook

Visualization & Application

Streamlit

interactive mapping

Google Street View

API & External Services

FastAPI

Pydantic

Geoapify

French geocoding service

Hugging Face Inference

Qwen

MLOps & Deployment

MLflow

Neon PostgreSQL

AWS S3

boto3

Docker

Hugging Face Spaces

Git

GitHub

Project Structure

6_french-property-intelligence/
│
├── api/
│   ├── generation.py
│   ├── geocoding.py
│   ├── environment.py
│   ├── main.py
│   ├── model_loader.py
│   └── schemas.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── deploy/
│   ├── api/
│   ├── mlflow/
│   └── streamlit/
│
├── notebooks/
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── reports/
│
├── scripts/
│
├── src/
│   └── inference.py
│
├── streamlit/
│   └── app.py
│
├── .gitignore
└── README.md

Generated data, production model binaries, credentials, caches, and other runtime artifacts are excluded from Git.

Reproducibility

1. Clone the Repository

git clone https://github.com/AlexandraBelj/6-french-property-intelligence.git
cd 6_french-property-intelligence

2. Create a Virtual Environment

python -m venv .venv

Activate it using the command appropriate for the operating system.

3. Install Dependencies

Install the project dependencies required for the workflow being reproduced.

Deployment-specific requirements are kept with the corresponding Docker deployment configuration.

4. Add the Source Data

The raw transaction data must be placed under:

data/raw/

Raw and processed datasets are excluded from Git because of their size and data-management requirements.

5. Run the FastAPI Service Locally

With the required environment variables configured:

uvicorn api.main:app --host 127.0.0.1 --port 8001

Interactive API documentation is then available locally at:

http://127.0.0.1:8001/docs

6. Run Streamlit Locally

Configure Streamlit to use the local API and run:

streamlit run streamlit/app.py

The deployed application is available at:

https://huggingface.co/spaces/AlexBelj/french-property-intelligence

7. Production Configuration

The deployed services require environment-specific configuration for resources such as:

AWS S3;

MLflow;

PostgreSQL;

Geoapify;

Hugging Face Inference.

Credentials and API tokens must be supplied through environment variables or deployment secrets and must never be committed to Git.

Privacy & External Services

The application relies on external services for some contextual features.

Depending on the requested feature, property-location information may be transmitted to services used for:

address geocoding;

nearby-place retrieval;

Street View exploration;

generative text generation.

In particular, the GenAI description service receives the resolved address together with structured property, valuation, and environment information required to construct the description.

No API keys, cloud credentials, or access tokens are stored in the repository.

Users should avoid entering unnecessary personal information into address or property fields.

Limitations

Statistical Valuation

Predictions are based on historical transaction data and do not constitute professional appraisals.

A limited set of property characteristics is available to the model.

Important price determinants such as exact condition, floor, exposure, view, renovation quality, energy performance, or detailed interior characteristics may be unavailable.

Model accuracy varies across property types and geographic markets.

Rare property/location combinations may have substantially greater uncertainty than typical observations.

Historical predictive performance does not guarantee equivalent performance on future transactions.

Geographic Context

Address geocoding can resolve to a street or approximate location rather than an exact building.

Nearby-place data depends on the coverage and current data available from external providers.

Street View availability and panorama position depend on Google coverage and may not correspond exactly to the entered property.

Generative AI

The generated property description is intended to summarize supplied structured information.

The prompt constrains unsupported property claims, but generated text should still be interpreted as automatically produced content.

GenAI does not modify the estimated property value.

Availability depends on the external inference provider.

External Services

Some application features depend on third-party APIs. Temporary provider outages or quota limits may affect environment, Street View, or description functionality without affecting the underlying valuation model.

Conclusion

French Property Intelligence demonstrates an end-to-end data-science workflow from large-scale transaction analysis to deployed machine-learning inference and an interactive property-intelligence product.

Two specialized LightGBM models are used for residential valuation:

the house model achieves a held-out MAE of €68,296, RMSE of €136,391, and R² of 0.702;

the apartment model achieves a held-out MAE of €47,499, RMSE of €104,031, and R² of 0.824.

The production architecture separates the main responsibilities of the system:

Machine Learning → property valuation
Geospatial APIs  → factual location context
GenAI            → natural-language presentation
FastAPI          → service layer
Streamlit        → user interface
MLflow           → experiment tracking
Neon             → tracking metadata
AWS S3           → model and artifact storage

The result is a reproducible and deployed property-valuation system that combines statistical prediction with useful geographic context while preserving a clear boundary between model-based valuation and AI-generated presentation