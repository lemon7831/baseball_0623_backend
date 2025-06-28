# 使用 Python 3.9 作為基礎映像
FROM python:3.9-slim

# 設定工作目錄
WORKDIR /app

# 複製所有檔案到容器內部
COPY . /app

# 安裝依賴套件
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 開啟 8000 port
EXPOSE 8000

# 啟動應用程式
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
