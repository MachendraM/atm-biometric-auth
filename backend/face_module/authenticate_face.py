import cv2
import numpy as np
import dlib
import pickle
from scipy.spatial.distance import cosine
from sqlalchemy.orm import Session
from app.models import User 
from .face_utils import get_models
from fastapi.responses import JSONResponse

detector, predictor, embedder = get_models()

THRESHOLD = 0.3

def preprocess_face(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32')
    return face

def align_face(img, face_rect):
    shape = predictor(img, face_rect)
    aligned_face = dlib.get_face_chip(img, shape, size=160)
    return aligned_face

def get_face_embedding(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) == 0:
        return None
    
    if len(faces) > 1:
        print("[WARN] Multiple faces detected. Only using the first.")

    aligned_face = align_face(frame, faces[0])
    face_pp = preprocess_face(aligned_face)
    embedding = embedder.embeddings([face_pp])[0]
    embedding = embedding / np.linalg.norm(embedding)

    return embedding

def authenticate_face_from_image(frame: np.ndarray, db: Session):
    embedding = get_face_embedding(frame)
    if embedding is None:
        return  {
            "success": False, 
            "message": "no face detected try again"
        }
    print(f"[DEBUG] uploaded image, Embedding Sample: {embedding[:5]}")

    users = db.query(User).all()
    best_match = None
    best_distance = float('inf')

    for user in users:
        db_embedding = pickle.loads(user.face_embedding)
        print(f"[DEBUG] User: {user.name}, Embedding Sample: {db_embedding[:5]}")
        distance = cosine(embedding, db_embedding)
        print(f"[DEBUG] Comparing with {user.name}: distance = {distance:.4f}")

        if distance < THRESHOLD and distance < best_distance:
            best_match = user
            best_distance = distance

    if best_match:
        print(f"[DEBUG] Best match: {best_match.name}, distance: {best_distance}")
        return  {
            "success": True,
            "message": f"Authentication successful. Welcome, {best_match.name}!",
            "name": best_match.name
        }
    else:
        return {
            "success": False, 
            "message": "Authentication failed."
        }