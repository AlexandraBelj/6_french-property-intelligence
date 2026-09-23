# ============================================================
# French Property Intelligence — Streamlit deployment
#
# PURPOSE:
# Run the Streamlit frontend as a Hugging Face Docker Space.
#
# Hugging Face Space layout:
#   Dockerfile
#   app.py
#   requirements.txt
#
# The Streamlit frontend calls the production FastAPI service.
# It does not contain the ML model or provider secrets.
# ============================================================

FROM python:3.12-slim

WORKDIR /app

# Install frontend dependencies.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit application.
COPY app.py ./app.py

# Hugging Face Docker Spaces expose port 7860.
EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true", "--browser.gatherUsageStats=false"]