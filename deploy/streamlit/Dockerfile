# ============================================================
# French Property Intelligence — Streamlit deployment
#
# PURPOSE:
# Run the Streamlit frontend as a Hugging Face Docker Space.
#
# The Streamlit application does NOT contain the ML model and
# does NOT access AWS or Geoapify directly.
#
# Production flow:
#
# Streamlit -> FastAPI Space -> model / geocoding / environment
# ============================================================

FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer.
COPY streamlit/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy only the Streamlit application.
COPY streamlit/app.py ./app.py

# Hugging Face Docker Spaces expect the application on port 7860.
EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=7860", \
     "--server.headless=true"]