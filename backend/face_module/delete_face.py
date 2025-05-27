from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.database import SessionLocal  # Adjust to your actual import path
from app.models import User             # Adjust to your actual import path

def delete_user_by_name(user_name: str) -> bool:
    """
    Deletes a user by their unique name from the database.
    Returns True if deletion was successful, False otherwise.
    """
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter(User.name == user_name).first()
        if not user:
            print(f"[WARN] No user found with the name {user_name}")
            return False

        db.delete(user)
        db.commit()
        print(f"[INFO] User with the name {user_name} deleted successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to delete user with name {user_name}: {e}")
        return False
    finally:
        db.close()