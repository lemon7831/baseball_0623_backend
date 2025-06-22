import math
import numpy as np

def angle_between(p1, p2, p3):
    """計算角度 ∠p2（p1-p2-p3）"""
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_pitching_biomechanics(skeleton_data: dict) -> dict:
    features = {
        'avg_elbow_angle': None,
        'avg_shoulder_slope_deg': None,
        'avg_hip_slope_deg': None,
        'avg_torso_twist_deg': None,
        'max_hand_speed_px_per_s': None,
        'max_stride_length_px': None,
        'min_elbow_height_px': None,
        'avg_head_elbow_dist_px': None,
        'avg_shoulder_width_px': None
    }

    fps = 30  # 若有更準確 fps 可替換
    prev_wrist = None
    prev_time = None

    elbow_angles = []
    shoulder_slopes = []
    hip_slopes = []
    torso_twists = []
    hand_speeds = []
    stride_lengths = []
    elbow_heights = []
    elbow_head_dists = []
    shoulder_widths = []

    for frame in skeleton_data.get("frames", []):
        preds_group = frame.get("predictions", [])
        if not preds_group or not preds_group[0]:
            continue
        person = preds_group[0][0]  # 取第一個人

        kp = person["keypoints"]
        scores = person["keypoint_scores"]

        def valid(idx):
            return idx < len(scores) and scores[idx] > 0.3

        # 1. 手肘角度（右手）
        if valid(6) and valid(8) and valid(10):
            angle = angle_between(kp[6], kp[8], kp[10])  # shoulder-elbow-wrist
            elbow_angles.append(angle)

        # 2. 肩膀水平角度
        if valid(5) and valid(6):
            dx = kp[6][0] - kp[5][0]
            dy = kp[6][1] - kp[5][1]
            shoulder_slopes.append(math.degrees(math.atan2(dy, dx)))

        # 3. 髖部水平角度
        if valid(11) and valid(12):
            dx = kp[12][0] - kp[11][0]
            dy = kp[12][1] - kp[11][1]
            hip_slopes.append(math.degrees(math.atan2(dy, dx)))

        # 4. 軀幹扭轉（右肩對左髖）
        if valid(6) and valid(11):
            dx = kp[11][0] - kp[6][0]
            dy = kp[11][1] - kp[6][1]
            torso_twists.append(math.degrees(math.atan2(dy, dx)))

        # 5. 投球手速度（像素/s）
        if valid(10):
            curr_wrist = np.array(kp[10])
            t = frame["frame_idx"] / fps
            if prev_wrist is not None and prev_time is not None:
                dist = np.linalg.norm(curr_wrist - prev_wrist)
                dt = t - prev_time
                if dt > 0:
                    hand_speeds.append(dist / dt)
            prev_wrist = curr_wrist
            prev_time = t

        # 6. 步幅：踝左右 x 距離
        if valid(15) and valid(16):
            dx = abs(kp[15][0] - kp[16][0])
            stride_lengths.append(dx)

        # 7. 手肘高度
        if valid(8):
            elbow_heights.append(kp[8][1])

        # 8. 頭 - 手肘距離
        if valid(0) and valid(8):
            dist = np.linalg.norm(np.array(kp[0]) - np.array(kp[8]))
            elbow_head_dists.append(dist)

        # 9. 肩膀寬度
        if valid(5) and valid(6):
            dist = abs(kp[6][0] - kp[5][0])
            shoulder_widths.append(dist)

    features['avg_elbow_angle'] = float(np.mean(elbow_angles)) if elbow_angles else None
    features['avg_shoulder_slope_deg'] = float(np.mean(shoulder_slopes)) if shoulder_slopes else None
    features['avg_hip_slope_deg'] = float(np.mean(hip_slopes)) if hip_slopes else None
    features['avg_torso_twist_deg'] = float(np.mean(torso_twists)) if torso_twists else None
    features['max_hand_speed_px_per_s'] = float(np.max(hand_speeds)) if hand_speeds else None
    features['max_stride_length_px'] = float(np.max(stride_lengths)) if stride_lengths else None
    features['min_elbow_height_px'] = float(np.min(elbow_heights)) if elbow_heights else None
    features['avg_head_elbow_dist_px'] = float(np.mean(elbow_head_dists)) if elbow_head_dists else None
    features['avg_shoulder_width_px'] = float(np.mean(shoulder_widths)) if shoulder_widths else None

    return features