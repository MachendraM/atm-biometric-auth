from sqlalchemy import Column, Integer, String, Float, LargeBinary
from app.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    face_embedding = Column(LargeBinary, nullable=False)  # We will store pickled numpy array
    image_path = Column(String, nullable=True)
    balance = Column(Float, default=0.0)
