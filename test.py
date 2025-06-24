import json
from KinematicsModule import extract_pitching_biomechanics as extract_v1
from KinematicsModulev2 import extract_pitching_biomechanics as extract_v2

# 讀入 JSON 姿勢資料
with open("CH_videos_4s_pitch_0001_skeleton.json", "r") as f:
    skeleton_data = json.load(f)

# 呼叫 v1 和 v2 的分析函數
result_v1 = extract_v1(skeleton_data)
result_v2 = extract_v2(skeleton_data)

# 顯示比對結果
print("\n===== V1 結果 =====")
for k, v in result_v1.items():
    print(f"{k}: {v}")

print("\n===== V2 結果 =====")
for k, v in result_v2.items():
    print(f"{k}: {v}")
