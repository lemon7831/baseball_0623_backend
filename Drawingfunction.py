import cv2
import os
import math
from typing import Tuple # 導入 Tuple 以正確標註回傳型別

# COCO 骨架連線對
COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # 頭臉
    (5, 6),                             # 肩膀
    (5, 7), (7, 9),                     # 左手臂
    (6, 8), (8, 10),                    # 右手臂
    (5, 11), (6, 12), (11, 12),         # 軀幹臀部
    (11, 13), (13, 15),                 # 左腿
    (12, 14), (14, 16)                  # 右腿
]

def render_video_with_pose_and_max_ball_speed(input_video_path: str,
                                              pose_json: dict,
                                              ball_json: dict,
                                              output_video_path: str,
                                              pixel_to_meter: float = 0.04,
                                              min_valid_speed_kmh: float = 30,
                                              max_valid_speed_kmh: float = 200) -> Tuple[str, float]:
    """
    同時渲染骨架與棒球框，顯示最大球速，並排除不合理速度。

    Args:
        input_video_path (str): 原始影片路徑
        pose_json (dict): 骨架偵測資料
        ball_json (dict): 棒球框資料
        output_video_path (str): 輸出影片路徑
        pixel_to_meter (float): 像素轉公尺的比例（例如18.44m/450px ≈ 0.04）
        min_valid_speed_kmh (float): 最小有效速度
        max_valid_speed_kmh (float): 最大有效速度

    Returns:
        tuple[str, float]: 輸出影片路徑 和 最大球速
    """
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'X264') # <-- 修改這一行
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 確保 pose_frames 和 ball_frames 被正確初始化
    pose_frames = {f['frame_idx']: f.get('predictions', []) for f in pose_json.get('frames', [])}
    ball_frames = {frame_idx: box for frame_idx, box in ball_json.get('results', [])}

    prev_center = None
    prev_frame_idx = None
    max_speed_kmh = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- 畫骨架 ---
        predictions_raw = pose_frames.get(frame_idx, [])
        # 根據 pose_json 的實際結構調整這裡的取值邏輯
        # 如果 predictions_raw 是一個包含列表的列表，取第一個列表
        # 否則直接使用 predictions_raw
        predictions = predictions_raw[0] if predictions_raw and isinstance(predictions_raw, list) and predictions_raw and isinstance(predictions_raw[0], list) else predictions_raw


        for person in predictions:
            keypoints = person.get('keypoints', [])
            scores = person.get('keypoint_scores', [])
            if not keypoints or not scores:
                continue

            for i, (x, y) in enumerate(keypoints):
                if i < len(scores) and scores[i] > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

            for (start, end) in COCO_CONNECTIONS:
                if start < len(keypoints) and end < len(keypoints):
                    x1, y1 = keypoints[start]
                    x2, y2 = keypoints[end]
                    if scores[start] > 0.3 and scores[end] > 0.3:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        # --- 畫棒球 + 計算速度 ---
        if frame_idx in ball_frames:
            current_ball_box = ball_frames[frame_idx]
            # 確保 current_ball_box 不是 None 才能進行 map(int, ...)
            if current_ball_box is not None:
                x1, y1, x2, y2 = map(int, current_ball_box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Baseball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if prev_center is not None and prev_frame_idx is not None:
                    dx = cx - prev_center[0]
                    dy = cy - prev_center[1]
                    distance_pixels = math.sqrt(dx**2 + dy**2)
                    dt = (frame_idx - prev_frame_idx) / fps

                    if dt > 0:
                        distance_m = distance_pixels * pixel_to_meter
                        speed_mps = distance_m / dt
                        speed_kmh = speed_mps * 3.6

                        if min_valid_speed_kmh <= speed_kmh <= max_valid_speed_kmh:
                            max_speed_kmh = max(max_speed_kmh, speed_kmh)

                prev_center = (cx, cy)
                prev_frame_idx = frame_idx
            # 如果 current_ball_box 是 None，則跳過棒球框繪製和速度計算

        # --- 畫最大球速 ---
        label = f"Max Speed: {max_speed_kmh:.1f} km/h"
        cv2.rectangle(frame, (30, 30), (360, 80), (0, 0, 0), -1)  # 黑底
        cv2.putText(frame, label, (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # 白字

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return output_video_path, max_speed_kmh