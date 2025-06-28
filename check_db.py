import psycopg2
from psycopg2 import sql
import pandas as pd

# 你的資料庫連線字串
DATABASE_URL = "postgresql://baseball_0623_postgres_user:5EcgbNjxL90WgsAWGU5xNylqYoEvNBWx@dpg-d1d1nrmmcj7s73fai6k0-a.oregon-postgres.render.com/baseball_0623_postgres"

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
        print("資料庫裡的資料表：")
        for schema, table in tables:
            print(f"{schema}.{table}")

        table_to_query = tables[0][1] if tables else None
        if table_to_query:
            print(f"\n資料表 {table_to_query} 的前5筆資料：")

            # 用 pandas 讀取查詢結果
            query = sql.SQL("SELECT * FROM {} LIMIT 100").format(sql.Identifier(table_to_query))
            df = pd.read_sql_query(query.as_string(conn), conn)
            print(df)
        else:
            print("找不到任何資料表")

    except Exception as e:
        print("查詢資料失敗:", e)

    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()
