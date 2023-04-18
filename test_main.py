"""
Run the API:
    - uvicorn main:app --reload
To run the tests:
    python -m pytest test_main.py
    pytest -vv test_main.py
See other branches:
    git log --graph --decorate --oneline
"""

from fastapi.testclient import TestClient
from main import app


# Instantiate the testing client with our app.
client = TestClient(app)

# Test "/" endpoint
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}


## Test for ML

def test_prediction_request():
    """
    POST method with "/model/predict/" endpoint.
    Check data types.
    """
    r = client.post(
        url="/model/predict/",
        json={
        "age":38,
        "workclass":"Private",
        "fnlgt":76878,
        "education":"11th",
        "education-num":7,
        "marital-status":"Married-civ-spouse",
        "occupation":"Craft-repair",
        "relationship":"Husband",
        "race":"White",
        "sex":"Male",
        "capital-gain":5178,
        "capital-loss":0,
        "hours-per-week":40,
        "native-country":"United-States"
    }
    )
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data['query']['age'], int)
    assert isinstance(data['query']['workclass'], str)
    assert isinstance(data['query']['fnlgt'], (int, float))
    assert isinstance(data['query']['education'], str)
    assert isinstance(data['query']['education-num'], int)
    assert isinstance(data['query']['marital-status'], str)
    assert isinstance(data['query']['occupation'], str)
    assert isinstance(data['query']['relationship'], str)
    assert isinstance(data['query']['race'], str)
    assert isinstance(data['query']['sex'], str)
    assert isinstance(data['query']['capital-gain'], (int, float))
    assert isinstance(data['query']['capital-loss'], (int, float))
    assert isinstance(data['query']['hours-per-week'], (int, float))
    assert isinstance(data['query']['native-country'], str)


def test_prediction_result():
    """
    POST method with "/model/predict/" endpoint.
    Check wheter the prediction is 1 or 0
    """
    r = client.post(
        url="/model/predict/",
        json={
        "age":38,
        "workclass":"Private",
        "fnlgt":76878,
        "education":"11th",
        "education-num":7,
        "marital-status":"Married-civ-spouse",
        "occupation":"Craft-repair",
        "relationship":"Husband",
        "race":"White",
        "sex":"Male",
        "capital-gain":5178,
        "capital-loss":0,
        "hours-per-week":40,
        "native-country":"United-States"
    }
    )
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data['prediction'], int)
    assert data['prediction'] in [1, 0]