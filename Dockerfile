FROM python:3.9

# 確保必要套件齊全，並添加 ffmpeg
RUN apt-get update && apt-get install -y \
    libpq-dev \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 複製所有檔案到容器中
COPY . /app

# 安裝 Python 套件
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 對外開放 8080 port
EXPOSE 8080

# 啟動 FastAPI 伺服器
CMD ["python", "main.py"]
