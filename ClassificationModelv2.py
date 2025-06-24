def classify_pitch_quality(features):
    """
    接受新版 biomechanical features 進行投球品質分類。
    可根據 domain knowledge 或模型來定義簡單規則或呼叫 ML 模型。
    """
    if features["Trunk_flexion_excursion"] is None:
        return 0  # 缺資料則給最低分

    score = 0
    if 50 <= features["Trunk_flexion_excursion"] <= 80:
        score += 1
    if -10 <= features["Pelvis_obliquity_at_FC"] <= 10:
        score += 1
    if 140 <= features["Shoulder_abduction_at_BR"] <= 160:
        score += 1
    if -80 <= features["Trunk_flexion_at_BR"] <= -40:
        score += 1
    return score  # 滿分 4
