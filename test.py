import json
from KinematicsModule import extract_pitching_biomechanics as extract_v1
from KinematicsModulev2 import extract_pitching_biomechanics as extract_v2
from ClassificationModel import classify_pitch_quality as classify_v1
from ClassificationModelv2 import classify_pitch_quality as classify_v2

# 讀入 JSON 姿勢資料
with open("CH_videos_4s_pitch_0001_skeleton.json", "r") as f:
    skeleton_data = json.load(f)

# 呼叫 v1 和 v2 的分析函數
result_v1 = extract_v1(skeleton_data)
result_v2 = extract_v2(skeleton_data)

# 儲存分析結果為 JSON 檔案
with open("pitch_0001_features_v1.json", "w") as f1:
    json.dump(result_v1, f1, indent=4)

with open("pitch_0001_features_v2.json", "w") as f2:
    json.dump(result_v2, f2, indent=4)

print("已成功儲存 result_v1 和 result_v2 到 JSON 檔案。")

# 顯示比對結果
print("\n===== V1 Biomechanics Features =====")
for k, v in result_v1.items():
    print(f"{k}: {v}")

print("\n===== V2 Biomechanics Features =====")
for k, v in result_v2.items():
    print(f"{k}: {v}")

# 分類評分
score_v1 = classify_v1(result_v1)
score_v2 = classify_v2(result_v2)

print("\n===== Classification Scores =====")
print(f"V1 Score: {score_v1}")
print(f"V2 Score: {score_v2}")
