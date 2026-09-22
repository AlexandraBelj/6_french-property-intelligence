FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["sh", "-c", "mlflow server \
    --backend-store-uri \"$DATABASE_URL\" \
    --artifacts-destination s3://french-property-intelligence-539152062780-eu-central-1-an/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 7860 \
    --allowed-hosts 'alexbelj-french-property-intelligence-mlflow.hf.space' \
    --x-frame-options NONE"]