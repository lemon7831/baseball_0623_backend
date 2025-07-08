import os

# GCS 設定
GCS_BUCKET_NAME = "baseball_cloud_storage"

# 外部 API 端點
POSE_API_URL = "https://mmpose-api-new-1069614647348.us-central1.run.app/pose_video"
BALL_API_URL = "https://base-ball-detect-api-1069614647348.us-east4.run.app/predict"

# Render PostgreSQL 資料庫 URL
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:baseball000@34.66.34.45:5432/postgres")


# LemonAPI
# POSE_API_URL = "http://localhost:8000/pose_video"
# BALL_API_URL = "http://localhost:8080/predict"
