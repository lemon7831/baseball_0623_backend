import os
import asyncio
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse # 導入 RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from typing import Dict, Any

# 導入 boto3
import boto3
from botocore.exceptions import ClientError # 用於處理 S3 錯誤

# 導入同目錄下的自定義模組
from Drawingfunction import render_video_with_pose_and_max_ball_speed
from KinematicsModule import extract_pitching_biomechanics
from ClassificationModel import classify_pitch_quality

# FastAPI 應用實例
app = FastAPI()

# CORS 設置，允許所有來源、方法、標頭，以處理跨域問題
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://baseball-0623-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 外部 API 端點
POSE_API_URL = "https://mmpose-api-924124779607.us-central1.run.app/pose_video"
BALL_API_URL = "https://baseball-api-gpu-669140972615.us-central1.run.app/predict-video"

# Render PostgreSQL 資料庫 URL
DATABASE_URL = "postgresql://baseball_database_1761_user:KuHDnfORZhnrd8LJlvomxARTPB3ekC93@dpg-d1but1gdl3ps73f3418g-a.oregon-postgres.render.com/baseball_database_1761"

# SQLAlchemy 設定
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 定義資料庫模型
class PitchAnalysis(Base):
    __tablename__ = "pitch_analyses"

    id = Column(Integer, primary_key=True, index=True)
    video_path = Column(String, index=True) # 這個欄位現在會儲存 S3 URL
    max_speed_kmh = Column(Float)
    pitch_score = Column(Integer)
    biomechanics_features = Column(JSON)

# 創建資料庫表 (如果不存在的話)
Base.metadata.create_all(bind=engine)

# S3 配置
# 從環境變數獲取 AWS 憑證和區域
AWS_ACCESS_KEY_ID = 'AKIAXPO3ZRZ4WVRP3MEM'
AWS_SECRET_ACCESS_KEY = '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'
AWS_REGION = 'ap-northeast-1'
S3_BUCKET_NAME = 'cji101-28'

# 初始化 S3 客戶端
s3_client = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and AWS_REGION and S3_BUCKET_NAME:
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        print("S3 client initialized successfully.")
    except Exception as e:
        print(f"Error initializing S3 client: {e}")
        s3_client = None # 確保 s3_client 為 None 如果初始化失敗
else:
    print("WARNING: AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME) not fully configured in environment variables. S3 upload will not work.")


# 輔助函數：上傳檔案到 S3
def upload_file_to_s3(file_path: str, bucket: str, object_name: str = None):
    """將檔案上傳到 S3 儲存桶"""
    if s3_client is None:
        raise Exception("S3 client not initialized. AWS credentials might be missing or incorrect.")

    if object_name is None:
        object_name = os.path.basename(file_path)

    try:
        # 使用 upload_file，它會自動處理檔案流
        s3_client.upload_file(file_path, bucket, object_name)
        # 構造 S3 公開 URL (假設你的 S3 Bucket 配置為公開讀取)
        s3_url = f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{object_name}"
        print(f"檔案 {file_path} 已成功上傳到 {s3_url}")
        return s3_url
    except ClientError as e:
        print(f"上傳檔案到 S3 失敗: {e}")
        raise HTTPException(status_code=500, detail=f"無法上傳影片到雲端儲存: {e}")
    except Exception as e:
        print(f"S3 上傳時發生未知錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"S3 上傳時發生未知錯誤: {e}")


@app.post("/analyze-pitch/")
async def analyze_pitch(video_file: UploadFile = File(...)):
    """
    接收棒球投球影片，進行運動學分析、投球評分，並渲染結果影片，
    最後將結果儲存至資料庫並回傳給前端。
    """
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="未上傳影片檔案")

    # 在 /tmp 目錄下創建臨時檔案，這是 Render 等 PaaS 平台的最佳實踐
    temp_video_path = os.path.join("/tmp", f"temp_upload_{video_file.filename}")
    local_output_video_path = os.path.join("/tmp", f"rendered_output_{video_file.filename}")

    # 確保 /tmp 目錄存在，雖然通常 Render 會自動提供
    os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(local_output_video_path), exist_ok=True) #

    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"無法儲存上傳的影片檔案: {e}")

    try:
        with open(temp_video_path, "rb") as f:
            video_bytes = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"無法讀取影片內容: {e}")

    pose_data = None
    ball_data = None

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # 修改點：將 "video_file" 更改為 "file" (根據你之前的指示)
            pose_task = client.post(POSE_API_URL, files={"file": (video_file.filename, video_bytes, "video/mp4")})
            ball_task = client.post(BALL_API_URL, files={"file": (video_file.filename, video_bytes, "video/mp4")})

            pose_response, ball_response = await asyncio.gather(pose_task, ball_task)

            pose_response.raise_for_status()
            ball_response.raise_for_status()

            pose_data = pose_response.json()
            ball_data = ball_response.json()

        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"外部 API 請求失敗: {e}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"外部 API 返回錯誤: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"處理外部 API 回應時發生錯誤: {e}")

    output_video_filename = f"rendered_{video_file.filename}"
    # 不再需要 output_videos 資料夾，因為影片將上傳到 S3
    # local_output_video_path 已經定義為 /tmp 下的路徑

    rendered_video_path, max_speed_kmh = render_video_with_pose_and_max_ball_speed(
        input_video_path=temp_video_path,
        pose_json=pose_data,
        ball_json=ball_data,
        output_video_path=local_output_video_path # 渲染到本地臨時路徑
    )

    # 將渲染後的影片上傳到 S3
    s3_video_url = None
    try:
        s3_video_url = upload_file_to_s3(local_output_video_path, S3_BUCKET_NAME, output_video_filename)
    except Exception as e:
        # 如果 S3 上傳失敗，仍然刪除臨時檔案並報錯
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(local_output_video_path):
            os.remove(local_output_video_path)
        raise HTTPException(status_code=500, detail=f"影片上傳到 S3 失敗: {e}")


    biomechanics_features = extract_pitching_biomechanics(pose_data)
    pitch_score = classify_pitch_quality(biomechanics_features)

    db = SessionLocal()
    new_analysis_id = None
    try:
        new_analysis = PitchAnalysis(
            video_path=s3_video_url, # 將 S3 URL 儲存到資料庫
            max_speed_kmh=max_speed_kmh,
            pitch_score=pitch_score,
            biomechanics_features=biomechanics_features
        )
        db.add(new_analysis)
        db.commit()
        db.refresh(new_analysis)
        new_analysis_id = new_analysis.id
        print(f"數據已成功保存到資料庫，ID: {new_analysis_id}")
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"資料庫儲存失敗: {e}")
    finally:
        db.close()

    # 刪除本地臨時檔案
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    if os.path.exists(local_output_video_path):
        os.remove(local_output_video_path)


    return JSONResponse(content={
        "message": "影片分析成功",
        "output_video_path": s3_video_url, # 返回 S3 URL 給前端
        "max_speed_kmh": round(max_speed_kmh, 2),
        "pitch_score": pitch_score,
        "biomechanics_features": biomechanics_features,
        "new_analysis_id": new_analysis_id
    })

# 修改 get_rendered_video 函數
# 這個接口現在將直接重定向到 S3 URL。
# 也可以選擇移除此接口，讓前端直接使用從 /analyze-pitch/ 獲取的 S3 URL 或從 /history/ 獲取的歷史紀錄中的 S3 URL。
# 如果保留，你可能需要從資料庫中根據 filename 查找對應的 S3 URL。
# 但由於 filename 並非唯一的 S3 object key，且前端會直接獲取 S3 URL，這個接口的最佳實踐是根據 analysis_id 來查找。
# 目前的實現會假設 filename 就是 S3 object name，並直接構造 URL。
@app.get("/output_videos/{filename}")
async def get_rendered_video(filename: str):
    # 這個接口的存在主要是為了兼容之前可能的前端調用方式
    # 在實際應用中，如果前端已經直接從 /analyze-pitch/ 或 /history/ 獲取了 S3 URL
    # 這個接口的使用頻率會大大降低，甚至可以移除。
    # 如果你希望通過這個接口來獲取歷史影片，你需要根據 filename 從資料庫中查找對應的 S3 URL。
    # 這裡只是一個示例，它假設 filename 可以直接用來構造 S3 URL。
    if not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and AWS_REGION and S3_BUCKET_NAME):
        raise HTTPException(status_code=500, detail="S3 配置不完整，無法獲取影片。")

    s3_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{filename}"
    return RedirectResponse(url=s3_url, status_code=302)


# 修改 get_history_analyses 函數
@app.get("/history/")
async def get_history_analyses():
    db = SessionLocal()
    try:
        history_records = db.query(PitchAnalysis).all()
        return [
            {
                "id": record.id,
                "video_path": record.video_path, # 這裡已經是 S3 URL
                "max_speed_kmh": record.max_speed_kmh,
                "pitch_score": record.pitch_score,
                "biomechanics_features": record.biomechanics_features
            }
            for record in history_records
        ]
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"無法獲取歷史紀錄: {e}")
    finally:
        db.close()
