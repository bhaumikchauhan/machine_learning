# The REST endpoints for cars and their trips: an APIRouter is a @RestController with a @RequestMapping prefix.
# carsharing.py registers it with app.include_router (forgetting that gives 404s for all of these).
#
# Where FastAPI takes each parameter from, decided by the function signature alone:
# - name appears in the path, like {car_id}   -> path parameter (@PathVariable); always required
# - simple type (int, str, ...) not in the path -> query parameter (@RequestParam): /api/cars?size=m&doors=4;
#                                                  required unless it has a default
# - a Pydantic/SQLModel class                  -> JSON request body (@RequestBody + @Valid)
# - Annotated[X, Depends(f)]                   -> injected: FastAPI calls f and passes the result (@Autowired)
# The type hints do the conversion and validation: doors=abc gives a 422 response, and the function never runs.
#
# def vs async def: these are plain `def` because SQLModel's session blocks while it waits for the database.
# FastAPI runs plain `def` endpoints in a thread pool, so a blocking call doesn't stall other requests. Use
# `async def` only when everything inside is awaited (async libraries). A model's predict() is blocking
# (CPU work), so a prediction endpoint should be plain `def` too.
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from db import get_session
from routers.auth import get_current_user
from schemas import Car, CarInput, CarOutput, Trip, TripInput, TripOutput, UserOutput

router = APIRouter(prefix="/api/cars", tags=["cars"])     # tags group the endpoints on the /docs page

# Type aliases so each endpoint doesn't repeat the Annotated[...] (the pattern the FastAPI docs recommend)
SessionDep = Annotated[Session, Depends(get_session)]
CurrentUser = Annotated[UserOutput, Depends(get_current_user)]


class BadTripException(Exception):
    # A domain exception; carsharing.py registers a handler that turns it into a 422 response,
    # like an @ExceptionHandler in a @ControllerAdvice
    pass


# The return type hint is also the response schema: FastAPI validates the return value against it and
# sends only those fields (the course said FastAPI ignores it; that changed in FastAPI 0.89).

@router.get("/")
def get_cars(session: SessionDep, size: str | None = None, doors: int | None = None) -> list[CarOutput]:
    # GET /api/cars, GET /api/cars?size=m, GET /api/cars?size=m&doors=4. An empty result is a 200 with [].
    query = select(Car)
    if size:
        query = query.where(Car.size == size)
    if doors:
        query = query.where(Car.doors >= doors)
    return list(session.exec(query).all())


@router.get("/{car_id}")
def car_by_id(car_id: int, session: SessionDep) -> CarOutput:
    car = session.get(Car, car_id)          # look up by primary key
    if car is None:
        # an id that doesn't exist is a 404, not an empty object; raising aborts the function
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No car with id={car_id}.")
    return car                              # a Car row; FastAPI converts it to CarOutput (including its trips)


@router.post("/", status_code=status.HTTP_201_CREATED)
def add_car(car_input: CarInput, session: SessionDep, user: CurrentUser) -> CarOutput:
    # Protected: the `user` dependency returns 401 without a valid token, before this body runs.
    # status_code in the decorator sets the success code (default 200; 201 Created is the REST convention).
    car = Car.model_validate(car_input)     # input schema -> table object (id still None)
    session.add(car)
    session.commit()                        # INSERT; nothing is saved without a commit
    session.refresh(car)                    # reload from the database, so car.id is filled in
    return car


@router.put("/{car_id}")
def change_car(car_id: int, new_data: CarInput, session: SessionDep) -> CarOutput:
    # a path parameter (which car) and a body (its new data) in one operation
    car = session.get(Car, car_id)
    if car is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No car with id={car_id}.")
    car.sqlmodel_update(new_data.model_dump())   # copy every input field onto the row (the id isn't in the input)
    session.commit()
    session.refresh(car)
    return car


@router.delete("/{car_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_car(car_id: int, session: SessionDep) -> None:
    # 204 No Content: success, nothing to send back
    car = session.get(Car, car_id)
    if car is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No car with id={car_id}.")
    session.delete(car)
    session.commit()


@router.post("/{car_id}/trips", status_code=status.HTTP_201_CREATED)
def add_trip(car_id: int, trip_input: TripInput, session: SessionDep) -> TripOutput:
    car = session.get(Car, car_id)
    if car is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No car with id={car_id}.")
    # model_validate with update= adds fields that aren't in the input (the foreign key)
    trip = Trip.model_validate(trip_input, update={"car_id": car_id})
    if trip.end < trip.start:
        raise BadTripException("Trip ends before it starts")
    car.trips.append(trip)                  # the relationship takes care of the insert
    session.commit()
    session.refresh(trip)
    return trip
