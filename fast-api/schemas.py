# The data classes. SQLModel = Pydantic (validation, JSON) + SQLAlchemy (ORM) in one class:
# - a class WITHOUT table=True is a schema only: a DTO for request or response bodies
# - a class WITH table=True is also a database table: an @Entity
# Fields are declared with type hints; Pydantic generates __init__, validation and type conversion
# ("5" becomes 5 for an int field, "five" raises a validation error, which FastAPI turns into a 422 response).
# A field with a default is optional; one without is required.
import bcrypt
from sqlmodel import SQLModel, Field, Relationship


# --- Trips: a car has many trips (one-to-many) ---

class TripInput(SQLModel):
    # what the client sends to create a trip: no id, the database makes that
    start: int
    end: int
    description: str


class TripOutput(TripInput):
    # what the API sends back: the input fields plus the id
    id: int


class Trip(TripInput, table=True):
    # the table. `int | None = None` + primary_key: None until saved, then the database fills it in
    id: int | None = Field(default=None, primary_key=True)
    car_id: int = Field(foreign_key="car.id")              # table names are the lower-case class names
    car: "Car" = Relationship(back_populates="trips")      # like @ManyToOne, both sides linked by back_populates


# --- Cars ---
# Separate input and output schemas are normal, not duplication: the input has no id (the client can't know it),
# and the output decides exactly what leaves the server, so a database field (like a password hash) can't leak.

class CarInput(SQLModel):
    size: str
    fuel: str | None = "electric"       # optional, with a default
    doors: int
    transmission: str | None = "auto"

    # example shown and prefilled in the /docs page ("Try it out"); doesn't affect validation
    model_config = {"json_schema_extra": {"examples": [{"size": "m", "doors": 5, "transmission": "manual",
                                                        "fuel": "hybrid"}]}}


class Car(CarInput, table=True):
    id: int | None = Field(default=None, primary_key=True)
    # like @OneToMany(mappedBy = "car", cascade = CascadeType.REMOVE): deleting a car deletes its trips. Without
    # cascade_delete, SQLAlchemy would set the trips' car_id to NULL on delete, which the column doesn't allow.
    trips: list[Trip] = Relationship(back_populates="car", cascade_delete=True)


class CarOutput(CarInput):
    id: int
    trips: list[TripOutput] = []        # nested models work out of the box, in and out


# --- Users, for the login in routers/auth.py ---

class UserOutput(SQLModel):
    # deliberately no password_hash: what an endpoint may expose about a user
    id: int
    username: str


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)        # unique constraint and index, like @Column(unique = true)
    password_hash: str = ""

    def set_password(self, password: str):
        # store only a salted bcrypt hash, never the password (the course used passlib, which is unmaintained)
        self.password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def verify_password(self, password: str) -> bool:
        return bcrypt.checkpw(password.encode(), self.password_hash.encode())
