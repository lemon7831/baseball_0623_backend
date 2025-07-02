def classify_pitch_quality(features):
    """
    接受新版 biomechanical features 進行投球品質分類。
    可根據 domain knowledge 或模型來定義簡單規則或呼叫 ML 模型。
    """
    # Initialize score
    score = 0

    # Safely get features, default to None if not present
    trunk_flexion_excursion = features.get("Trunk_flexion_excursion")
    pelvis_obliquity_at_fc = features.get("Pelvis_obliquity_at_FC")
    shoulder_abduction_at_br = features.get("Shoulder_abduction_at_BR")
    trunk_flexion_at_br = features.get("Trunk_flexion_at_BR")

    # If the primary feature is missing, return 0
    if trunk_flexion_excursion is None:
        return 0

    # Apply scoring logic, checking for None before comparison
    if trunk_flexion_excursion is not None and 50 <= trunk_flexion_excursion <= 80:
        score += 1
    if pelvis_obliquity_at_fc is not None and -10 <= pelvis_obliquity_at_fc <= 10:
        score += 1
    if shoulder_abduction_at_br is not None and 140 <= shoulder_abduction_at_br <= 160:
        score += 1
    if trunk_flexion_at_br is not None and -80 <= trunk_flexion_at_br <= -40:
        score += 1

    return score  # 滿分 4
