from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base

DATABASE_URL = "postgresql://baseball_0623_postgres_user:5EcgbNjxL90WgsAWGU5xNylqYoEvNBWx@dpg-d1d1nrmmcj7s73fai6k0-a.oregon-postgres.render.com/baseball_0623_postgres"

Base = declarative_base()

from sqlalchemy import Column, Integer, String, Float, JSON

class PitchAnalysis(Base):
    __tablename__ = "pitch_analyses"

    id = Column(Integer, primary_key=True, index=True)
    video_path = Column(String, index=True)
    max_speed_kmh = Column(Float)
    pitch_score = Column(Integer)
    biomechanics_features = Column(JSON)
    ball_score = Column(Float)

def reset_database():
    engine = create_engine(DATABASE_URL)
    metadata = MetaData()

    # 反射資料庫結構，綁定 engine
    metadata.reflect(bind=engine)

    # 刪除所有資料表
    metadata.drop_all(bind=engine)
    print("✅ 資料庫所有資料表已刪除")

    # 重新建立 ORM 定義的表
    Base.metadata.create_all(bind=engine)
    print("✅ 資料表已重新建立")

if __name__ == "__main__":
    reset_database()
