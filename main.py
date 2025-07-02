import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
import uvicorn
from config import GCS_BUCKET_NAME
from database import get_db, PitchAnalysis
from models import PitchAnalysisUpdate
from crud import get_pitch_analysis, get_pitch_analyses, create_pitch_analysis, update_pitch_analysis, delete_pitch_analysis
from services import analyze_pitch_service

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

@app.post("/analyze-pitch/")
async def analyze_pitch(video_file: UploadFile = File(...), pitcher_name: str = Form(...), db: Session = Depends(get_db)):
    """
    接收棒球投球影片，進行運動學分析、投球評分，並渲染結果影片，
    最後將結果儲存至資料庫並回傳給前端。
    """
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="未上傳影片檔案")

    try:
        analysis_result = await analyze_pitch_service(video_file, pitcher_name)

        # 儲存至資料庫
        new_analysis = create_pitch_analysis(
            db=db,
            video_path=analysis_result["output_video_url"],
            pitcher_name=analysis_result["pitcher_name"],
            max_speed_kmh=analysis_result["max_speed_kmh"],
            pitch_score=analysis_result["pitch_score"],
            biomechanics_features=analysis_result["biomechanics_features"],
            ball_score=analysis_result["ball_score"]
        )
        logger.info(f"數據已成功保存到資料庫，ID: {new_analysis.id}")

        # 回傳結果
        return JSONResponse(content={
            "message": "影片分析成功",
            "output_video_url": new_analysis.video_path,
            "max_speed_kmh": round(new_analysis.max_speed_kmh, 2),
            "pitch_score": new_analysis.pitch_score,
            "ball_score": new_analysis.ball_score,
            "biomechanics_features": new_analysis.biomechanics_features,
            "new_analysis_id": new_analysis.id,
            "pitcher_name": new_analysis.pitcher_name
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"影片分析處理失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"影片分析處理失敗: {e}")


@app.get("/history/")
async def get_history_analyses(pitcher_name: str = None, db: Session = Depends(get_db)):
    try:
        history_records = get_pitch_analyses(db, pitcher_name)
        return [
            {
                "id": record.id,
                "video_path": record.video_path,
                "max_speed_kmh": record.max_speed_kmh,
                "pitch_score": record.pitch_score,
                "ball_score": record.ball_score,
                "biomechanics_features": record.biomechanics_features,
                "pitcher_name": record.pitcher_name
            }
            for record in history_records
        ]
    except SQLAlchemyError as e:
        logger.error(f"無法獲取歷史紀錄: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"無法獲取歷史紀錄: {e}")


@app.delete("/analyses/{analysis_id}")
async def delete_analysis(analysis_id: int, db: Session = Depends(get_db)):
    try:
        if not delete_pitch_analysis(db, analysis_id):
            raise HTTPException(status_code=404, detail="分析紀錄未找到")
        logger.info(f"分析紀錄 ID: {analysis_id} 已成功刪除")
        return {"message": "分析紀錄已成功刪除"}
    except SQLAlchemyError as e:
        logger.error(f"刪除分析紀錄失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"刪除分析紀錄失敗: {e}")

@app.put("/analyses/{analysis_id}")
async def update_analysis(analysis_id: int, updated_data: PitchAnalysisUpdate, db: Session = Depends(get_db)):
    try:
        analysis = update_pitch_analysis(db, analysis_id, updated_data)
        if not analysis:
            raise HTTPException(status_code=404, detail="分析紀錄未找到")
        logger.info(f"分析紀錄 ID: {analysis_id} 已成功更新")
        return analysis
    except SQLAlchemyError as e:
        logger.error(f"更新分析紀錄失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"更新分析紀錄失敗: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)