# face_utils.py
import dlib
from keras_facenet import FaceNet

print("[INFO] Loading FaceNet and dlib models...")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\M SURYA PANDI\Desktop\ATM\backend\face_module\shape_predictor_68_face_landmarks.dat")
embedder = FaceNet()  # Load only once

print("[INFO] Models loaded successfully.")

def get_models():
    return detector, predictor, embedder
