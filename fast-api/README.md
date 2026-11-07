# FastAPI, condensed

The car-sharing API from Pluralsight's *FastAPI Foundations* (Reindert-Jan Ekker), condensed for someone who knows Spring Boot. The course builds it over seven modules; this is the end state, rewritten, with comments that explain each idea where it's used. Read the files in this order:

1. [carsharing.py](carsharing.py): the app, startup (`lifespan`), routers, CORS, middleware, exception handler, how to run it
2. [schemas.py](schemas.py): Pydantic/SQLModel classes; input vs output schemas, tables, relationships
3. [routers/cars.py](routers/cars.py): the endpoints; where each parameter comes from; status codes; `def` vs `async def`
4. [db.py](db.py): the engine and the per-request session (a dependency with `yield`)
5. [routers/auth.py](routers/auth.py): OAuth2 password flow, a protected endpoint
6. [tests/conftest.py](tests/conftest.py), [tests/test_cars.py](tests/test_cars.py): `TestClient`, `dependency_overrides`, a mock session

## Run it

```
venv\Scripts\activate
pip install -r requirements.txt
python create_user.py          # a user for logging in
fastapi dev carsharing.py      # then open http://localhost:8000/docs
pytest                         # 8 tests, in-memory database
```

In `/docs`, click **Authorize** and log in to try the protected `POST /api/cars/`.

## Spring Boot to FastAPI

| Spring Boot | FastAPI | Where |
|---|---|---|
| `@SpringBootApplication`, embedded Tomcat | `app = FastAPI()`, run by uvicorn | carsharing.py |
| `@RestController` + `@RequestMapping("/api/cars")` | `APIRouter(prefix="/api/cars")` + `app.include_router` | routers/cars.py |
| `@GetMapping("/{id}")` | `@router.get("/{car_id}")` | routers/cars.py |
| `@PathVariable`, `@RequestParam(required = false)` | function parameters, by name and default | routers/cars.py |
| `@RequestBody @Valid CarDto` | a Pydantic parameter: validated, 422 if invalid | routers/cars.py |
| DTO vs `@Entity` | schema class vs `table=True` class | schemas.py |
| `@OneToMany(mappedBy, cascade = REMOVE)` | `Relationship(back_populates, cascade_delete=True)` | schemas.py |
| `@Autowired`, request scope | `Annotated[X, Depends(f)]`; `yield` for cleanup | db.py |
| `@ResponseStatus(CREATED)`, `ResponseStatusException` | `status_code=` in the decorator, `raise HTTPException` | routers/cars.py |
| `@ControllerAdvice` / `@ExceptionHandler` | `@app.exception_handler(...)` | carsharing.py |
| Servlet `Filter` | `@app.middleware("http")` | carsharing.py |
| `@PostConstruct` / `@PreDestroy` | `lifespan` | carsharing.py |
| Spring Security + `Principal` | `OAuth2PasswordBearer` + `Depends(get_current_user)` | routers/auth.py |
| springdoc / Swagger UI | built in: `/docs`, `/redoc`, `/openapi.json` | |
| `MockMvc`, `@MockBean` | `TestClient`, `app.dependency_overrides` | tests/ |
| Flyway | Alembic (not used here; `create_all` makes missing tables) | |

## Left out from the course, and why

- **Setup and tooling:** installing Python, PyCharm/VS Code debugging (set a breakpoint and run `carsharing.py` directly; the `__main__` block allows that), Postman (it can import `/openapi.json`).
- **The JSON-file "database"** (modules 4 and 5): only a stepping stone to SQLModel.
- **HTML pages, Jinja templates, form posts and cookies** (module 7): server-rendered web pages, not needed for a JSON API or a model endpoint.
- **HTTP Basic authentication** (module 8): the course replaces it with OAuth2 itself.

## Changed from the course

- **The return type hint is the response schema.** The course says FastAPI ignores return types; since FastAPI 0.89 it uses them to validate and filter the response.
- **`passlib` replaced by `bcrypt`**: passlib is unmaintained and breaks with current bcrypt versions.
- **Deleting a car with trips failed** (`NOT NULL constraint failed: trip.car_id`): fixed with `cascade_delete=True`, and tested.
- **`POST` returns 201 Created** instead of 200, the REST convention.
- **Tests use an in-memory database** through `dependency_overrides`, instead of the real `carsharing.db`.
- **The 404 message** in `add_trip` printed Python's built-in `id` function; it now uses `car_id`.

## Deployment (module 9, in short)

- `fastapi run carsharing.py` in production (no auto-reload); `fastapi dev` only during development.
- **Behind a reverse proxy** (nginx): it listens on ports 80/443, forwards to uvicorn on 8000, so the app doesn't run as root, and it absorbs some attacks.
- **HTTPS is required**: cloud hosts usually provide it; on your own server, free certificates from Let's Encrypt (certbot).
- **Other options:** a platform that hosts it for you, or a Docker container (phase 1 of the plan).

## For the ML model API

- **Load the model once in `lifespan`**, not per request.
- **The prediction endpoint is a plain `def`**: `predict()` is blocking CPU work, which FastAPI runs in its thread pool. Inside an `async def` it would stall every other request.
- **Input and output schemas** give the model a validated contract: the features as a Pydantic class, the prediction and probability as another.
- **Middleware** is the natural place to log latency and requests (prediction logging).
