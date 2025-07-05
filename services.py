import os
import shutil
import httpx
import asyncio
import joblib
import logging
from sqlalchemy.orm import Session
from config import GCS_BUCKET_NAME, POSE_API_URL, BALL_API_URL
from gcs_utils import upload_video_to_gcs
from Drawingfunction import render_video_with_pose_and_max_ball_speed, save_specific_frames
from KinematicsModulev2 import extract_pitching_biomechanics
from ClassificationModelv2 import classify_pitch_quality
from BallClassification import classify_ball_quality
from typing import Dict, Optional, Tuple
import crud
logger = logging.getLogger(__name__)
API_TIMEOUT = 300

# 載入球路預測模型 用來分類好壞球
ball_prediction_model = joblib.load('random_forest_model.pkl')

# 取得比較模型 輸入資料庫 比較對象 球路 返回比較標準模型
def get_comparison_model(db: Session, benchmark_player_name: str, detected_pitch_type: str):
    """
    【修改後的輔助函式】
    透過 crud.py 智慧地從資料庫中尋找最適合的比對模型。
    """
    profile_model = None
    
    # 1. 如果偵測到的球種有效，優先嘗試尋找最精準的模型 (投手 + 球種)
    if detected_pitch_type and detected_pitch_type != "Unknown":
        # 組合出與 buildModel.py 完全一致的模型名稱
        ideal_model_name = f"{benchmark_player_name}_{detected_pitch_type}_v1"
        logger.info(f"服務層：正在嘗試載入球種專屬模型: {ideal_model_name}")
        profile_model = crud.get_pitch_model_by_name(db, model_name=ideal_model_name)
        if profile_model:
            return profile_model

    # 2. 如果找不到專屬模型，或球種未知，則嘗試尋找該投手的「通用」模型作為備案
    fallback_model_name = f"{benchmark_player_name}_all_v1"
    logger.warning(f"找不到或未指定專屬模型，嘗試載入通用模型: {fallback_model_name}")
    profile_model = crud.get_pitch_model_by_name(db, model_name=fallback_model_name)

    return profile_model

# 分析生物力學特徵函數 輸入影片 返回 運動力學特徵 骨架
async def analyze_video_kinematics(video_bytes: bytes, filename: str) -> Tuple[Dict, Dict]:
    """
    呼叫 Pose API，計算生物力學特徵，並同時回傳原始 pose_data 以供畫圖使用。
    """
    logger.info("服務層：(子任務) 正在呼叫 POSE API...")
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        files = {"file": (filename, video_bytes, "video/mp4")}
        response = await client.post(POSE_API_URL, files=files)
        response.raise_for_status()
        pose_data = response.json()
    logger.info("服務層：(子任務) 正在計算生物力學特徵...")
    biomechanics_features = extract_pitching_biomechanics(pose_data)
    return biomechanics_features, pose_data

# 評分函數 輸入運動力學特徵 評判標準資料 輸出分數
def calculate_score_from_comparison(features: dict, profile_data: dict) -> int:
    """
    【新的評分函式】
    根據使用者的特徵與標準模型的差距，計算出一個 0-100 的分數。
    差距越小，分數越高。
    """
    if not profile_data:
        return 0 # 如果沒有模型可以比對，分數為 0

    total_score = 0
    feature_count = 0

    for key, user_value in features.items():
        # 只比對在模型中有定義的特徵
        profile_stats = profile_data.get(key.lower())
        if not profile_stats or user_value is None:
            continue
        
        mean = profile_stats.get('mean')
        std = profile_stats.get('std')

        # 確保模型中有 mean 和 std，且 std 不為 0
        if mean is None or std is None or std == 0:
            continue

        # 計算 Z-score，代表偏離了幾個標準差
        z_score = abs((user_value - mean) / std)
        
        # 將 Z-score 轉換為 0-100 的分數
        # 這裡使用一個簡單的轉換：Z-score 為 0 (完全符合平均) 得 100 分
        # Z-score 每增加 1 (偏離一個標準差)，就扣 25 分 (可調整)
        # 最低為 0 分
        feature_score = max(0, 100 - z_score * 25)
        
        total_score += feature_score
        feature_count += 1
    
    if feature_count == 0:
        return 0

    # 回傳所有特徵的平均分數
    final_score = int(total_score / feature_count)
    return final_score

# 棒球軌跡分析函數 輸入影片輸出球路軌跡
async def analyze_ball_flight(video_bytes: bytes, filename: str) -> Dict:
    """
    呼叫 Ball API 以獲取球路相關數據。
    """
    logger.info("服務層：(子任務) 正在呼叫 BALL API...")
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        files = {"file": (filename, video_bytes, "video/mp4")}
        response = await client.post(BALL_API_URL, files=files)
        response.raise_for_status()
        return response.json()

# 主要分析路由 輸入資料庫 影片 球員名稱 比較對象 返回分析結果
async def analyze_pitch_service(
        db,
        video_file, 
        player_name,
        benchmark_player_name=None
        ):
    
    # 步驟 1 嘗試暫存原始影片
    temp_video_path = f"temp_{video_file.filename}"
    
    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
    except Exception as e:
        logger.error(f"無法儲存影片檔案: {e}", exc_info=True)
        raise e

    try:
        with open(temp_video_path, "rb") as f:
            video_bytes = f.read()
    except Exception as e:
        logger.error(f"無法讀取影片內容: {e}", exc_info=True)
        raise e

    # 步驟 2: 並行呼叫 API 分析骨架跟球路
    (kinematics_results, ball_data) = await asyncio.gather(
            analyze_video_kinematics(video_bytes, video_file.filename),
            analyze_ball_flight(video_bytes, video_file.filename)
            )
    
    # 從kinematics_results拿出骨架資料跟運動力學特徵
    biomechanics_features, pose_data = kinematics_results
    
    # 從球路資料拿到pitch_type
    detected_pitch_type = ball_data.get("predicted_pitch_type",None)
    
    # 決定投手分數的比較標準
    comparison_target_name = benchmark_player_name if benchmark_player_name else player_name
    profile_model = get_comparison_model(db, comparison_target_name, detected_pitch_type)

    # 利用比對標準和運動力學特徵來計算投手分數
    pitch_score = 0
    profile_data_for_frontend = None
    if profile_model:
        profile_data_for_frontend = profile_model.profile_data
        pitch_score = calculate_score_from_comparison(
                features=biomechanics_features,
                profile_data=profile_data_for_frontend
            )
    else:
        logger.warning(f"在資料庫中找不到任何可用的比對模型 (比對對象: {comparison_target_name})，pitch_score 將設為 0。")

    # 計算投球分數
    ball_score = classify_ball_quality(ball_data, ball_prediction_model)
        
    # 渲染影片
    try:
        rendered_video_local_path, max_speed_kmh = render_video_with_pose_and_max_ball_speed(
            input_video_path=temp_video_path,
            pose_json=pose_data,
            ball_json=ball_data
        )
    except Exception as e:
        logger.error(f"影片渲染失敗: {e}", exc_info=True)
        raise e

    # 上傳至 GCS
    gcs_video_url = None
    try:
        destination_blob_name = f"render_videos/rendered_{video_file.filename}"
        gcs_video_url = upload_video_to_gcs(
            bucket_name=GCS_BUCKET_NAME,
            source_file_path=rendered_video_local_path,
            destination_blob_name=destination_blob_name
        )
    except Exception as e:
        logger.error(f"GCS 上傳失敗: {e}", exc_info=True)
        raise e

    # 儲存關鍵影格圖片
    release_frame_url = None
    landing_frame_url = None
    shoulder_frame_url = None
    frame_indices = {
        "release": biomechanics_features.get("release_frame"),
        "landing": biomechanics_features.get("landing_frame"),
        "shoulder": biomechanics_features.get("shoulder_frame")
        }
        
    saved_frame_paths = save_specific_frames(temp_video_path, frame_indices)

    # 上傳關鍵影格圖片到 GCS
    try:
        if "release_frame_path" in saved_frame_paths:
            release_frame_url = upload_video_to_gcs(
                bucket_name=GCS_BUCKET_NAME,
                source_file_path=saved_frame_paths["release_frame_path"],
                destination_blob_name=f"key_frames/release_{os.path.basename(saved_frame_paths['release_frame_path'])}"
            )
        if "landing_frame_path" in saved_frame_paths:
            landing_frame_url = upload_video_to_gcs(
                bucket_name=GCS_BUCKET_NAME,
                source_file_path=saved_frame_paths["landing_frame_path"],
                destination_blob_name=f"key_frames/landing_{os.path.basename(saved_frame_paths['landing_frame_path'])}"
            )
        if "shoulder_frame_path" in saved_frame_paths:
            shoulder_frame_url = upload_video_to_gcs(
                bucket_name=GCS_BUCKET_NAME,
                source_file_path=saved_frame_paths["shoulder_frame_path"],
                destination_blob_name=f"key_frames/shoulder_{os.path.basename(saved_frame_paths['shoulder_frame_path'])}"
            )
    except Exception as e:
        logger.error(f"GCS 上傳失敗: {e}", exc_info=True)
        raise e

    # 清理本地臨時檔案
    try:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(rendered_video_local_path):
            os.remove(rendered_video_local_path)
        for key in ["release_frame_path", "landing_frame_path", "shoulder_frame_path"]:
            path = saved_frame_paths.get(key)
            if path and os.path.exists(path):
                os.remove(path)
    except Exception as e:
        logger.warning(f"刪除暫存影片失敗: {e}", exc_info=True)

    # 返回分析結果
    return {
        "output_video_url": gcs_video_url,
        "max_speed_kmh": max_speed_kmh,
        "pitch_score": pitch_score,
        "ball_score": ball_score,
        "detected_pitch_type": detected_pitch_type,
        "biomechanics_features": biomechanics_features,
        "pitcher_name": player_name,
        "release_frame_url": release_frame_url,
        "landing_frame_url": landing_frame_url,
        "shoulder_frame_url": shoulder_frame_url,
        "model_profile": {
            "model_name": profile_model.model_name if profile_model else "N/A",
            "profile_data": profile_data_for_frontend
            },
    }
