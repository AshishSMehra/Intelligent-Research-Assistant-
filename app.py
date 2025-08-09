"""
Minimal FastAPI + JWT example (single file)
------------------------------------------
Run:
  pip install "fastapi[standard]" python-jose[cryptography] python-multipart
  uvicorn jwt_auth_minimal_fastapi:app --reload

Test:
  1) POST /auth/login with form fields username=admin password=admin123
  2) Copy access_token from response
  3) GET /admin/metrics with header: Authorization: Bearer <token>
"""
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import jwt, JWTError

from logging_config import logger

# -----------------------------------------------------------------------------
# Settings (demo values). Use env vars in real apps.
# -----------------------------------------------------------------------------
SECRET_KEY = "CHANGE_THIS_SUPER_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class User(BaseModel):
    username: str
    full_name: Optional[str] = None
    role: str = "user"
    disabled: bool = False

class TokenPayload(BaseModel):
    sub: str  # username
    role: str
    exp: int

# -----------------------------------------------------------------------------
# Fake user store (replace with DB later)
# -----------------------------------------------------------------------------
FAKE_USERS_DB = {
    "admin": {"username": "admin", "full_name": "Admin User", "password": "admin123", "role": "admin", "disabled": False},
    "ashish": {"username": "ashish", "full_name": "Ashish Mehra", "password": "password", "role": "user", "disabled": False},
}

# OAuth2 scheme extracts Bearer token from Authorization header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def authenticate_user(username: str, password: str) -> Optional[User]:
    rec = FAKE_USERS_DB.get(username)
    if not rec or rec["password"] != password:
        logger.warning(f"Authentication failed for user: {username}")
        return None
    logger.info(f"User '{username}' authenticated successfully.")
    return User(**{k: v for k, v in rec.items() if k != "password"})


def create_access_token(*, username: str, role: str, expires: Optional[timedelta] = None) -> str:
    exp = datetime.utcnow() + (expires or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {"sub": username, "role": role, "exp": int(exp.timestamp())}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    creds_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        data = TokenPayload(**payload)
    except JWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise creds_exc
    rec = FAKE_USERS_DB.get(data.sub)
    if not rec:
        logger.warning(f"User from token not found in DB: {data.sub}")
        raise creds_exc
    user = User(**{k: v for k, v in rec.items() if k != "password"})
    if user.disabled:
        logger.warning(f"Attempt to use inactive user account: {user.username}")
        raise HTTPException(status_code=400, detail="Inactive user")
    return user


async def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        logger.error(f"User '{user.username}' with role '{user.role}' attempted to access admin route.")
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user

# -----------------------------------------------------------------------------
# App & routes
# -----------------------------------------------------------------------------
app = FastAPI(title="JWT Minimal App", description="Single-file demo for JWT auth with FastAPI")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/auth/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(username=user.username, role=user.role)
    return Token(access_token=token)

@app.get("/admin/metrics")
async def metrics(_: User = Depends(require_admin)):
    return JSONResponse(status_code=status.HTTP_501_NOT_IMPLEMENTED, content={"message": "Not implemented"})
