# Run from the fast-api folder: pytest
# pytest collects files named test_*.py and functions named test_*; a plain `assert` is the check
# (on failure pytest shows both sides of the comparison).
from unittest.mock import Mock

from fastapi.testclient import TestClient

from carsharing import app
from routers.cars import add_car
from schemas import Car, CarInput, UserOutput


def test_add_and_get_car(client):
    response = client.post("/api/cars/", json={"size": "xl", "doors": 7})
    assert response.status_code == 201
    car = response.json()                       # the JSON body as Python dicts and lists
    assert car["doors"] == 7
    assert car["fuel"] == "electric"            # the default from CarInput

    response = client.get(f"/api/cars/{car['id']}")
    assert response.status_code == 200
    assert response.json()["size"] == "xl"


def test_filter_by_query_parameters(client):
    for size, doors in [("s", 3), ("m", 5), ("m", 3)]:
        client.post("/api/cars/", json={"size": size, "doors": doors})
    cars = client.get("/api/cars/", params={"size": "m", "doors": 4}).json()
    assert [(c["size"], c["doors"]) for c in cars] == [("m", 5)]


def test_unknown_car_is_404(client):
    response = client.get("/api/cars/999")
    assert response.status_code == 404
    assert response.json() == {"detail": "No car with id=999."}


def test_invalid_input_is_422(client):
    # validation happens before the endpoint runs: a missing field and a wrong type
    assert client.post("/api/cars/", json={"size": "m"}).status_code == 422
    assert client.get("/api/cars/", params={"doors": "four"}).status_code == 422


def test_trip_ending_before_start_uses_the_exception_handler(client):
    car = client.post("/api/cars/", json={"size": "m", "doors": 5}).json()
    response = client.post(f"/api/cars/{car['id']}/trips", json={"start": 100, "end": 50, "description": "back"})
    assert response.status_code == 422
    assert response.json() == {"detail": "Trip ends before it starts"}


def test_delete_car_with_trips(client):
    car = client.post("/api/cars/", json={"size": "m", "doors": 5}).json()
    client.post(f"/api/cars/{car['id']}/trips", json={"start": 0, "end": 12, "description": "shop"})
    assert client.delete(f"/api/cars/{car['id']}").status_code == 204
    assert client.get(f"/api/cars/{car['id']}").status_code == 404


def test_add_car_needs_a_token():
    # no fixture: the real get_current_user runs, and without an Authorization header it's a 401
    # (it fails before the database is touched)
    response = TestClient(app).post("/api/cars/", json={"size": "m", "doors": 5})
    assert response.status_code == 401


def test_add_car_with_mock_session():
    # A unit test of the function itself, without FastAPI: call it directly and pass a Mock as the session.
    # The Mock accepts any call and records it, so the test checks which database calls the function made.
    session = Mock()
    result = add_car(car_input=CarInput(size="xl", doors=2), session=session,
                     user=UserOutput(id=1, username="tester"))
    session.add.assert_called_once()
    session.commit.assert_called_once()
    assert isinstance(result, Car) and result.doors == 2
