# Login with OAuth2's "password flow": the client posts username + password (as a form) to /auth/token,
# gets a token back, and sends it on later requests as the header `Authorization: Bearer <token>`.
# In /docs, the "Authorize" button does this for you.
#
# WARNING, demo only (as in the course): the token here is just the username, so anyone can send any username
# and be logged in. A real app returns a signed token with an expiry (a JWT, e.g. with PyJWT) and checks the
# signature in get_current_user. The structure below stays the same.
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlmodel import Session, select

from db import get_session
from schemas import User, UserOutput

router = APIRouter(prefix="/auth", tags=["auth"])

# Reads the token from the Authorization header (401 if it's missing). tokenUrl tells /docs where to log in.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def get_current_user(token: Annotated[str, Depends(oauth2_scheme)],
                     session: Annotated[Session, Depends(get_session)]) -> UserOutput:
    # A dependency that itself has dependencies: FastAPI resolves the whole chain (token, then session).
    # An endpoint that declares `user: Annotated[UserOutput, Depends(get_current_user)]` is protected:
    # like a Spring Security filter plus an injected Principal.
    user = session.exec(select(User).where(User.username == token)).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token",
                            headers={"WWW-Authenticate": "Bearer"})
    return UserOutput.model_validate(user)     # convert the table row into the safe output schema


@router.post("/token")
def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
          session: Annotated[Session, Depends(get_session)]):
    # Depends() with no argument: FastAPI builds the class itself, here from the form fields username and password
    user = session.exec(select(User).where(User.username == form_data.username)).first()
    if user is None or not user.verify_password(form_data.password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect username or password")
    return {"access_token": user.username, "token_type": "bearer"}     # the shape OAuth2 requires
