# Creates a user for logging in (there's no sign-up endpoint). Run: python create_user.py
from getpass import getpass

from sqlmodel import Session, SQLModel

from db import engine
from schemas import User

if __name__ == "__main__":
    SQLModel.metadata.create_all(engine)       # the tables, in case the app hasn't run yet
    user = User(username=input("Username: "))
    user.set_password(getpass("Password: "))   # getpass doesn't echo the password
    with Session(engine) as session:
        session.add(user)
        session.commit()
    print(f"Created user {user.username}")
