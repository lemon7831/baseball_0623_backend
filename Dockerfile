# 使用基於 Debian 的 miniconda 映像，以便可以使用 conda
FROM continuumio/miniconda3:latest

# 確保必要套件齊全，並添加 ffmpeg
# 注意：minconda 映像可能已經包含一些基礎工具
RUN apt-get update && apt-get install -y \
    libpq-dev \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 複製所有檔案到容器中
COPY . /app

# 使用 conda 安裝 OpenCV
# 這將確保 OpenCV 及其相關依賴項以正確的方式安裝，以解決編解碼器問題
RUN conda install -c conda-forge opencv -y

# 安裝 Python 套件
# 確保您的 requirements.txt 中沒有 opencv-python 或 opencv-contrib-python
# 因為 OpenCV 已經透過 conda 安裝
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 對外開放 8080 port
EXPOSE 8080

# 啟動 FastAPI 伺服器
CMD ["python", "main.py"]

##[ERROR:0@0.493] global /io/opencv/modules/videoio/src/cap_ffmpeg_impl.hpp (2927) open Could not find encoder for codec_id=27, error: Encoder not found
##[ERROR:0@0.493] global /io/opencv/modules/videoio/src/cap_ffmpeg_impl.hpp (3002) open VIDEOIO/FFMPEG: Failed to initialize VideoWriter
##事實證明，由於許可問題，透過 pip 安裝 OpenCV 無法存取 AVC1 編解碼器。如果您改為從原始碼安裝，則可以解決此問題。
##透過 pip 卸載 OpenCV ( pip uninstall opencv-python opencv-contrib-python)
##使用 conda 安裝 OpenCV ( conda install -c conda-forge opencv)
