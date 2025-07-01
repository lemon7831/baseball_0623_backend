from google.cloud import storage

def upload_video_to_gcs(bucket_name, source_file_path, destination_blob_name):
    # 初始化 GCS 客戶端
    client = storage.Client.from_service_account_json("service-account-key.json")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # 上傳影片
    blob.upload_from_filename(source_file_path)
    print(f"影片已上傳至 gs://{bucket_name}/{destination_blob_name}")

    # 設定檔案為公開
    blob.make_public()
    print("影片已設為公開")

    # 回傳公開 URL
    public_url = blob.public_url
    print(f"公開網址：{public_url}")
    return public_url

upload_video_to_gcs(
    bucket_name="baseball_cloud_storage",
    source_file_path="pitch_0002.mp4",
    destination_blob_name="videos/my_video.mp4"
)


修改main.py和Drawingfunction.py 我不打算把渲染後的影片放在本地端"output_videos"資料夾我打算放在 GCS 
作法你可以參考GCS_test.py這個檔案 destination_blob_name設置為 render_videos/render_{filename}.mp4