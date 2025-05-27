from fastapi import APIRouter, UploadFile, File, HTTPException,Form,Depends
import numpy as np
import cv2
import pickle
from face_module.authenticate_face import authenticate_face_from_image
from face_module.register_face import get_face_embedding_from_image
from face_module.delete_face import delete_user_by_name
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User

face_routes = APIRouter()
@face_routes.post("/authenticate")
async def authenticate_user(image: UploadFile = File(...), db: Session = Depends(get_db)):
    img_bytes = await image.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    result = authenticate_face_from_image(frame, db)
    return {"message": result}


@face_routes.post("/register")
async def register_user(
    name: str = Form(...),
    balance: float = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Read uploaded image
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_np is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Extract face embedding
    face_embedding = get_face_embedding_from_image(img_np)
    if face_embedding is None:
        raise HTTPException(status_code=400, detail="No face detected in image")
    print("[DEBUG] New user embedding sample:", face_embedding[:5])


    serialized_embedding = pickle.dumps(face_embedding)

    # Check for duplicate name
    existing_user = db.query(User).filter(User.name == name).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User with this name already exists")

    # Store to database
    user = User(
        name=name,
        balance=balance,
        face_embedding=serialized_embedding,
        image_path=None  # Optionally store image file path here if needed
    )
    
    try:
        db.add(user)
        db.commit()
        db.refresh(user)

        db_embedding = pickle.loads(user.face_embedding)
        diff = np.linalg.norm(db_embedding - face_embedding)
        print(f"[DEBUG] Difference between saved and original embedding: {diff}")

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Database error")
    
    cv2.imwrite(f"debug_{name}.jpg", img_np)  # Save the actual uploaded image
    print(f"[DEBUG] Image saved as debug_{name}.jpg")
    print(f"[DEBUG] Embedding Sample: {face_embedding[:5]}")
    print(f"[DEBUG] Registering '{name}' - Embedding norm: {np.linalg.norm(face_embedding):.4f}")


    return {"message": f"User '{user.name}' registered successfully"}

@face_routes.delete("/delete")
async def delete_face(user_name: str = Form(...)):
    success = delete_user_by_name(user_name)
    if success:
        return {"message": f"Deleted registered face for user '{user_name}'"}
    else:
        raise HTTPException(status_code=404, detail=f"No registered face found with user name '{user_name}'")

@face_routes.get("/users")
async def get_all_users(db: Session = Depends(get_db)):
    print("✅ /face/users endpoint hit")
    users = db.query( User.name).all()
    return [user.name for user in users]

@face_routes.get("/users/{name}")
async def get_user_balance(name: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.name == name).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"balance": user.balance}

@face_routes.post("/users/withdraw")
async def withdraw_money(payload: dict, db: Session = Depends(get_db)):
    name = payload.get("name")
    amount = float(payload.get("amount"))
    user = db.query(User).filter(User.name == name).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.balance < amount:
        return {"success": False, "message": "Insufficient balance"}

    user.balance -= amount
    db.commit()
    return {"success": True, "new_balance": user.balance}

@face_routes.post("/users/Deposit")
async def deposit_money(payload: dict, db: Session = Depends(get_db)):
    name = payload.get("name")
    amount = float(payload.get("amount"))

    if name is None:
        raise HTTPException(status_code=400, detail="Name is required")
    if amount is None:
        raise HTTPException(status_code=400, detail="Amount is required")

    try:
        amount = float(amount)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid amount format")

    user = db.query(User).filter(User.name == name).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.balance += amount
    db.commit()
    return {"success": True, "new_balance": user.balance}