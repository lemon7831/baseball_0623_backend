def classify_pitch_quality(features: dict) -> str:
    """
    根據運動力學特徵，分類投球為「好球」或「壞球」。
    可以根據門檻值規則微調。

    Args:
        features (dict): 投球特徵字典（來自 biomechanics 特徵提取器）

    Returns:
        str: "Good" or "Bad"
    """

    score = 0

    # 1. 肘角
    if 80 <= features['avg_elbow_angle'] <= 120:
        score += 1

    # 2. 肩膀傾斜角
    if 80 <= features['avg_shoulder_slope_deg'] <= 100:
        score += 1

    # 3. 髖部傾斜角
    if 70 <= features['avg_hip_slope_deg'] <= 100:
        score += 1

    # 4. 軀幹扭轉角
    if 40 <= features['avg_torso_twist_deg'] <= 90:
        score += 1

    # 5. 最大手速
    if 1000 <= features['max_hand_speed_px_per_s'] <= 5000:
        score += 1

    # 6. 步幅
    if features['max_stride_length_px'] >= 100:
        score += 1

    # 7. 手肘高度
    if features['min_elbow_height_px'] > 200:
        score += 1

    # 8. 頭與肘的距離
    if 50 <= features['avg_head_elbow_dist_px'] <= 120:
        score += 1

    # 9. 肩膀展開
    if features['avg_shoulder_width_px'] > 40:
        score += 1

    # 給一個門檻（至少 6/9 符合才算好）
    return score