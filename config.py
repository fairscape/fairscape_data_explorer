# config.py
import os

FAIRSCAPE_BASE_URL = os.getenv("FAIRSCAPE_URL", "http://localhost:8080/api")
MINIO_ENDPOINT_URL = os.getenv("MINIO_URL", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "miniotestadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "miniotestsecret")
MINIO_DEFAULT_BUCKET = os.getenv("MINIO_BUCKET", "default")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")