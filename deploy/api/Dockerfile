# PURPOSE
# -------
# Production container for the French Property Intelligence FastAPI.
#
# The Hugging Face Space repository will contain:
# - api/ : HTTP API, geocoding and model loading
# - src/ : frozen production inference class
#
# model.pkl is NOT baked into the image.
# It is downloaded from the private S3 artifact store at application startup.

FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Application code copied from the Hugging Face Space repository.
COPY api ./api
COPY src ./src

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]