import psycopg2
from psycopg2 import sql
import pandas as pd
import os

# 從 main.py 導入 DATABASE_URL
# 為了避免循環導入和簡化，直接在這裡定義或從環境變量獲取
# 這裡假設 DATABASE_URL 已經在 main.py 中定義，並且可以安全地複製過來
# 或者，更好的做法是從環境變量中讀取
DATABASE_URL = "postgresql://postgres:baseball000@34.66.34.45:5432/postgres"

def main():
    # 建立連線
    try:
        conn = psycopg2.connect(DATABASE_URL)
        print("成功連接到資料庫")
    except Exception as e:
        print("連接資料庫失敗:", e)
        return

    cur = conn.cursor()

    try:
        # 查詢所有資料表
        cur.execute("""
            SELECT table_schema, table_name 
            FROM information_schema.tables 
            WHERE table_type='BASE TABLE' AND table_schema NOT IN ('pg_catalog', 'information_schema');
        """)
        tables = cur.fetchall()
        print("\n資料庫裡的資料表：")
        if not tables:
            print("找不到任何資料表")
            return

        for schema, table_name in tables:
            print(f"\n--- 資料表: {schema}.{table_name} 的前100資料 ---")
            try:
                # 用 pandas 讀取查詢結果
                query = sql.SQL("SELECT * FROM {}.{} LIMIT 100").format(
                    sql.Identifier(schema),
                    sql.Identifier(table_name)
                )
                df = pd.read_sql_query(query.as_string(conn), conn)
                if df.empty:
                    print("(此表格沒有資料)")
                else:
                    print(df)
            except Exception as e:
                print(f"查詢資料表 {schema}.{table_name} 失敗: {e}")

    except Exception as e:
        print("查詢資料表列表失敗:", e)

    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()