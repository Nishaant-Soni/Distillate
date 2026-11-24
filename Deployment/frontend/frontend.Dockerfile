# The demo app's image (work plan v5, section 6): Streamlit on Cloud Run's $PORT.
#
#   docker build -f frontend.Dockerfile -t distillate-frontend .      # from Deployment/frontend/
#   docker run -p 8080:8080 distillate-frontend                         # the mock, no settings
#
# No secrets in the image: the MLflow password and the other settings come from the
# environment (on Cloud Run, the password from Secret Manager with --set-secrets).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py presets.json ./
COPY demo_app/ demo_app/

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8080
# Headless, no usage statistics sent; exec so Streamlit gets Cloud Run's stop signal.
CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
