from sqlalchemy.orm import Session
from database import PitchAnalysis
from models import PitchAnalysisUpdate

def get_pitch_analysis(db: Session, analysis_id: int):
    return db.query(PitchAnalysis).filter(PitchAnalysis.id == analysis_id).first()

def get_pitch_analyses(db: Session, pitcher_name: str = None):
    query = db.query(PitchAnalysis)
    if pitcher_name:
        query = query.filter(PitchAnalysis.pitcher_name == pitcher_name)
    return query.all()

def create_pitch_analysis(db: Session, video_path: str, pitcher_name: str, max_speed_kmh: float, pitch_score: int, biomechanics_features: dict, ball_score: float, release_frame_url: str = None, landing_frame_url: str = None, shoulder_frame_url: str = None):
    db_analysis = PitchAnalysis(
        video_path=video_path,
        pitcher_name=pitcher_name,
        max_speed_kmh=max_speed_kmh,
        pitch_score=pitch_score,
        biomechanics_features=biomechanics_features,
        ball_score=ball_score,
        release_frame_url=release_frame_url,
        landing_frame_url=landing_frame_url,
        shoulder_frame_url=shoulder_frame_url
    )
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    return db_analysis

def update_pitch_analysis(db: Session, analysis_id: int, updated_data: PitchAnalysisUpdate):
    db_analysis = db.query(PitchAnalysis).filter(PitchAnalysis.id == analysis_id).first()
    if db_analysis:
        for field, value in updated_data.dict(exclude_unset=True).items():
            setattr(db_analysis, field, value)
        db.commit()
        db.refresh(db_analysis)
    return db_analysis

def delete_pitch_analysis(db: Session, analysis_id: int):
    db_analysis = db.query(PitchAnalysis).filter(PitchAnalysis.id == analysis_id).first()
    if db_analysis:
        db.delete(db_analysis)
        db.commit()
        return True
    return False
