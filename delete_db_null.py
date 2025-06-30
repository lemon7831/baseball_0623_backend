from sqlalchemy import create_engine, text

# 資料庫連線
DATABASE_URL = "postgresql://baseball_0623_postgres_user:5EcgbNjxL90WgsAWGU5xNylqYoEvNBWx@dpg-d1d1nrmmcj7s73fai6k0-a.oregon-postgres.render.com/baseball_0623_postgres"
engine = create_engine(DATABASE_URL)

# 使用 transaction，確保會 commit
with engine.begin() as conn:
    # 刪除 ball_score 為 NULL 的 row
    delete_query = """
    DELETE FROM pitch_analyses
    WHERE ball_score IS NULL;
    """
    result = conn.execute(text(delete_query))
    print("已刪除 ball_score 為 NULL 的資料")

# 驗證刪除是否成功
with engine.connect() as conn:
    check_query = """
    SELECT id, ball_score
    FROM pitch_analyses
    WHERE ball_score IS NULL;
    """
    result = conn.execute(text(check_query)).fetchall()
    print(f"剩下 {len(result)} 筆 ball_score 為 NULL 的資料：")
    for row in result:
        print(dict(row._mapping))
