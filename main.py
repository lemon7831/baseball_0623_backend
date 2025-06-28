import os
import asyncio
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from typing import Dict, Any
from fastapi.staticfiles import StaticFiles
import logging

# 導入同目錄下的自定義模組
from Drawingfunction import render_video_with_pose_and_max_ball_speed
from KinematicsModulev2 import extract_pitching_biomechanics
from ClassificationModelv2 import classify_pitch_quality
from BallClassification import classify_ball_quality
import joblib

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# FastAPI 應用實例
app = FastAPI()

# CORS 設置，允許所有來源、方法、標頭，以處理跨域問題
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 外部 API 端點
POSE_API_URL = "https://mmpose-api-1069614647348.us-central1.run.app/pose_video"
BALL_API_URL = "https://base-ball-detect-api-1069614647348.us-central1.run.app/predict"

# Render PostgreSQL 資料庫 URL
DATABASE_URL = "postgresql://baseball_0623_postgres_user:5EcgbNjxL90WgsAWGU5xNylqYoEvNBWx@dpg-d1d1nrmmcj7s73fai6k0-a.oregon-postgres.render.com/baseball_0623_postgres"

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
    video_path = Column(String, index=True)
    max_speed_kmh = Column(Float)
    pitch_score = Column(Integer)
    biomechanics_features = Column(JSON)
    ball_score = Column(Float)

# 創建資料庫表 (如果不存在的話)
Base.metadata.create_all(bind=engine)

# 載入球路預測模型
ball_prediction_model = joblib.load('random_forest_model.pkl')

@app.post("/analyze-pitch/")
async def analyze_pitch(video_file: UploadFile = File(...)):
    """
    接收棒球投球影片，進行運動學分析、投球評分，並渲染結果影片，
    最後將結果儲存至資料庫並回傳給前端。
    """
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="未上傳影片檔案")

    temp_video_path = f"temp_{video_file.filename}"
    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
    except Exception as e:
        logger.error(f"無法儲存影片檔案: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"無法儲存影片檔案: {e}")

    try:
        with open(temp_video_path, "rb") as f:
            video_bytes = f.read()
    except Exception as e:
        logger.error(f"無法讀取影片內容: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"無法讀取影片內容: {e}")

    pose_data = None
    ball_data = None

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            pose_task = client.post(POSE_API_URL, files={"file": (video_file.filename, video_bytes, "video/mp4")})
            ball_task = client.post(BALL_API_URL, files={"file": (video_file.filename, video_bytes, "video/mp4")})

            pose_response, ball_response = await asyncio.gather(pose_task, ball_task)

            pose_response.raise_for_status()
            ball_response.raise_for_status()

            pose_data = pose_response.json()
            ball_data = ball_response.json()

        except httpx.RequestError as e:
            logger.error(f"外部 API 請求失敗: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"外部 API 請求失敗: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"外部 API 返回錯誤: {e.response.text}", exc_info=True)
            raise HTTPException(status_code=e.response.status_code, detail=f"外部 API 返回錯誤: {e.response.text}")
        except Exception as e:
            logger.error(f"處理外部 API 回應時發生錯誤: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"處理外部 API 回應時發生錯誤: {e}")

    output_video_filename = f"rendered_{video_file.filename}"
    output_video_path = os.path.join("output_videos", output_video_filename)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    try:
        rendered_video_path, max_speed_kmh = render_video_with_pose_and_max_ball_speed(
            input_video_path=temp_video_path,
            pose_json=pose_data,
            ball_json=ball_data,
            output_video_path=output_video_path
        )
    except Exception as e:
        logger.error(f"影片渲染失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"影片渲染失敗: {e}")

    try:
        biomechanics_features = extract_pitching_biomechanics(pose_data)
        pitch_score = classify_pitch_quality(biomechanics_features)
    except Exception as e:
        logger.error(f"運動學分析或投球評分失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"運動學分析或投球評分失敗: {e}")

    try:
        ball_score = classify_ball_quality(ball_data, ball_prediction_model)
    except Exception as e:
        logger.error(f"球路分類失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"球路分類失敗: {e}")

    db = SessionLocal()
    new_analysis_id = None
    try:
        new_analysis = PitchAnalysis(
            video_path=rendered_video_path,
            max_speed_kmh=max_speed_kmh,
            pitch_score=pitch_score,
            biomechanics_features=biomechanics_features,
            ball_score=ball_score
        )
        db.add(new_analysis)
        db.commit()
        db.refresh(new_analysis)
        new_analysis_id = new_analysis.id
        logger.info(f"數據已成功保存到資料庫，ID: {new_analysis_id}")
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"資料庫儲存失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"資料庫儲存失敗: {e}")
    finally:
        db.close()

    try:
        os.remove(temp_video_path)
    except Exception as e:
        logger.warning(f"刪除暫存影片失敗: {e}", exc_info=True)

    return JSONResponse(content={
        "message": "影片分析成功",
        "output_video_path": rendered_video_path,
        "max_speed_kmh": round(max_speed_kmh, 2),
        "pitch_score": pitch_score,
        "ball_score": ball_score,
        "biomechanics_features": biomechanics_features,
        "new_analysis_id": new_analysis_id
    })

@app.get("/output_videos/{filename}")
async def get_rendered_video(filename: str):
    file_path = os.path.join("output_videos", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4")
    else:
        logger.error(f"影片未找到: {filename}")
        raise HTTPException(status_code=404, detail="影片未找到")

@app.get("/history/")
async def get_history_analyses():
    db = SessionLocal()
    try:
        history_records = db.query(PitchAnalysis).all()
        return [
            {
                "id": record.id,
                "video_path": record.video_path,
                "max_speed_kmh": record.max_speed_kmh,
                "pitch_score": record.pitch_score,
                "ball_score": record.ball_score,
                "biomechanics_features": record.biomechanics_features
            }
            for record in history_records
        ]
    except SQLAlchemyError as e:
        logger.error(f"無法獲取歷史紀錄: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"無法獲取歷史紀錄: {e}")
    finally:
        db.close()

os.makedirs("output_videos", exist_ok=True)
app.mount("/output_videos", StaticFiles(directory="output_videos"), name="output_videos")
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
