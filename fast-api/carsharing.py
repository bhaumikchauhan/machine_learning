# The application: the FastAPI object, startup, routers and app-wide behaviour (the @SpringBootApplication class).
#
# Run during development (reloads on code changes):   fastapi dev carsharing.py
# Run in production (no reload):                       fastapi run carsharing.py
# Then: http://localhost:8000/docs (Swagger UI, generated from the type hints; "Try it out" sends real requests),
# /redoc (the same, another layout), /openapi.json (the machine-readable spec: import it into Postman, or
# generate clients from it).
#
# Control flow: nothing runs top to bottom per request. The server (uvicorn) waits for requests, FastAPI matches
# each one to a path operation and calls that function.
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlmodel import SQLModel

from db import engine
from routers import auth, cars
from routers.cars import BadTripException


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code before `yield` runs once at startup, after `yield` once at shutdown (like @PostConstruct / @PreDestroy).
    # Here: create the tables that don't exist yet (no migrations; for those, use Alembic, the Flyway of Python).
    # For an ML API, this is where the model is loaded: once, not on every request (it can take seconds, a
    # prediction milliseconds). Keep it in a module-level variable or on app.state.
    SQLModel.metadata.create_all(engine)
    yield


app = FastAPI(title="Car Sharing", lifespan=lifespan)     # the title shows in /docs and in Postman
app.include_router(cars.router)
app.include_router(auth.router)

# CORS: a browser blocks JavaScript on another origin (e.g. a frontend on localhost:8080) from calling this API
# unless the API allows that origin. Only browsers enforce it: curl, Postman and other servers aren't affected.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    # Middleware wraps every request (like a servlet Filter): code before call_next runs before the endpoint,
    # code after it on the response. Here: how long the request took, as a response header. For an ML API this
    # is the natural place to log latency.
    start = time.perf_counter()
    response = await call_next(request)          # runs the matching endpoint
    response.headers["X-Process-Time"] = f"{time.perf_counter() - start:.4f}"
    return response


@app.exception_handler(BadTripException)
async def bad_trip_handler(request: Request, exc: BadTripException):
    # One central handler for a domain exception raised anywhere (like @ControllerAdvice + @ExceptionHandler).
    # The two parameters are required even when unused.
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": str(exc)})


if __name__ == "__main__":
    # Lets you run or debug the file directly from the IDE (breakpoints work), instead of `fastapi dev`
    uvicorn.run("carsharing:app", reload=True)
