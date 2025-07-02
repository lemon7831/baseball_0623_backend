import requests
import time
url = "https://base-ball-detect-api-1069614647348.us-east4.run.app/predict"
file_path = "pitch_0002.mp4"  # 替換成你自己的影片檔案路徑

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "video/mp4")}
    s = time.time()
    response = requests.post(url, files=files)
    e = time.time()
print(e-s)
print("Status Code:", response.status_code)
#print("Response:", response.text)
