import cv2
import dlib
import numpy as np
from .face_utils import get_models
import os

detector, predictor, embedder = get_models()

def preprocess_face(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32')
    return face

def align_face(img, face_rect):
    shape = predictor(img, face_rect)
    aligned_face = dlib.get_face_chip(img, shape, size=160)
    return aligned_face


def get_face_embedding_from_image(frame: np.ndarray) -> np.ndarray | None:
    """
    Extracts and returns the face embedding from the provided image frame.
    Returns None if no face is detected.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) == 0:
        return None

    if len(faces) > 1:
        print("[WARN] Multiple faces detected. Only using the first.")

    aligned_face = align_face(frame, faces[0])
    preprocessed = preprocess_face(aligned_face)
    embedding = embedder.embeddings([preprocessed])[0]
    embedding = embedding / np.linalg.norm(embedding)


    return embedding
