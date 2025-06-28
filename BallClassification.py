import pandas as pd
import numpy as np
'''
這個函數ball_json就是棒球api回傳的json檔案
model就是一個隨機森林模型
輸出浮點數代表是好球的機率
'''
def classify_ball_quality(ball_json,model,target_length=239):
    results = ball_json['results']
    x_list = []
    y_list = []
    for item in results:
        coords = item[1]
        x1, y1, x2, y2 = coords
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        x_list.append(center_x)
        y_list.append(center_y)
    # 補 0 到指定長度
    while len(x_list) < target_length:
        x_list.append(center_x)
        y_list.append(center_y)
    # 截斷到指定長度（保險）
    x_list = x_list[:target_length]
    y_list = y_list[:target_length]
    # 建立欄位名稱
    columns = [f'x_{i}' for i in range(target_length)] + [f'y_{i}' for i in range(target_length)]
    values = x_list + y_list
    df = pd.DataFrame([values], columns=columns)
    return float(model.predict_proba(df)[0][0])