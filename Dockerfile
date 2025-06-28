# 使用基於 Debian 的 miniconda 映像，以便可以使用 conda
FROM continuumio/miniconda3:latest

# 確保必要套件齊全，並添加 ffmpeg 和 libgl1
# libpq-dev (PostgreSQL 開發庫) 對於透過 Conda 安裝 psycopg2 通常不是必需的，
# 因為 Conda 套件通常會自行處理其C語言依賴。但如果仍遇到問題，可以取消註釋此行。
RUN apt-get update && apt-get install -y \
    libgl1 \
    ffmpeg \
    # libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 設定工作目錄
WORKDIR /app

# 複製 Conda 環境檔案到容器中
COPY environment.yml /app/environment.yml

# 創建 Conda 環境並安裝所有套件
# Conda 將處理所有的 Python 依賴，包括 OpenCV 和 psycopg2
# 這一步會從 environment.yml 中定義的 channels 安裝所有內容
RUN conda env create -f environment.yml

# 複製所有應用程式檔案到容器中
# 這是為了讓 Docker 能夠利用前一步的快取層
COPY . /app

# 對外開放 8080 port
EXPOSE 8080

# 啟動 FastAPI 伺服器
# 使用 'conda run -n <環境名稱>' 來確保應用程式在正確的 Conda 環境中執行
CMD ["conda", "run", "-n", "my_app_env", "python", "main.py"]
