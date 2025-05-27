# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import face_routes
from app.database import engine, Base
from app.models import User

Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS config to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include face recognition routes
app.include_router(face_routes, prefix="/face")


@app.get("/")
def root():
    return {"message": "Face Authentication ATM Backend is Running!"}
