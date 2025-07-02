import os
import shutil
import httpx
import asyncio
import joblib
import logging

from config import GCS_BUCKET_NAME, POSE_API_URL, BALL_API_URL
from gcs_utils import upload_video_to_gcs
from Drawingfunction import render_video_with_pose_and_max_ball_speed, save_specific_frames
from KinematicsModulev2 import extract_pitching_biomechanics
from ClassificationModelv2 import classify_pitch_quality
from BallClassification import classify_ball_quality

logger = logging.getLogger(__name__)

# 載入球路預測模型
ball_prediction_model = joblib.load('random_forest_model.pkl')

async def analyze_pitch_service(video_file, pitcher_name: str):
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
            raise e
        except httpx.HTTPStatusError as e:
            logger.error(f"外部 API 返回錯誤: {e.response.text}", exc_info=True)
            raise e
        except Exception as e:
            logger.error(f"處理外部 API 回應時發生錯誤: {e}", exc_info=True)
            raise e

    # 4. 渲染影片
    try:
        rendered_video_local_path, max_speed_kmh = render_video_with_pose_and_max_ball_speed(
            input_video_path=temp_video_path,
            pose_json=pose_data,
            ball_json=ball_data
        )
    except Exception as e:
        logger.error(f"影片渲染失敗: {e}", exc_info=True)
        raise e

    # 5. 上傳至 GCS
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

    # 6. 運動學分析與評分
    release_frame_url = None
    landing_frame_url = None
    shoulder_frame_url = None
    try:
        biomechanics_features = extract_pitching_biomechanics(pose_data)
        pitch_score = classify_pitch_quality(biomechanics_features)
        ball_score = classify_ball_quality(ball_data, ball_prediction_model)

        # 儲存關鍵影格圖片
        frame_indices = {
            "release": biomechanics_features.get("release_frame"),
            "landing": biomechanics_features.get("landing_frame"),
            "shoulder": biomechanics_features.get("shoulder_frame")
        }
        
        saved_frame_paths = save_specific_frames(temp_video_path, frame_indices)

        # 上傳關鍵影格圖片到 GCS
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
        logger.error(f"分析或評分失敗: {e}", exc_info=True)
        raise e

    # 7. 清理本地臨時檔案
    try:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(rendered_video_local_path):
            os.remove(rendered_video_local_path)
    except Exception as e:
        logger.warning(f"刪除暫存影片失敗: {e}", exc_info=True)

    return {
        "output_video_url": gcs_video_url,
        "max_speed_kmh": max_speed_kmh,
        "pitch_score": pitch_score,
        "ball_score": ball_score,
        "biomechanics_features": biomechanics_features,
        "pitcher_name": pitcher_name,
        "release_frame_url": release_frame_url,
        "landing_frame_url": landing_frame_url,
        "shoulder_frame_url": shoulder_frame_url
    }
