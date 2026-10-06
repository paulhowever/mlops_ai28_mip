FROM ghcr.io/mlflow/mlflow:v3.1.1

RUN pip install --no-cache-dir psycopg2-binary==2.9.10 boto3==1.35.99

EXPOSE 5000
