"""
Client API.

To run the API:
    - uvicorn main:app --reload
To run the client:
    - python client.py 
To run the tests:
    python -m pytest test_main.py
    pytest -vv test_main.py

Data to test the API

Data Examples ("salary":"<=50K"):
[
    {
        "age":53,
        "workclass":"Self-emp-not-inc",
        "fnlgt":168539,
        "education":"9th",
        "education-num":5,
        "marital-status":"Married-civ-spouse",
        "occupation":"Farming-fishing",
        "relationship":"Husband",
        "race":"White",
        "sex":"Male",
        "capital-gain":0,
        "capital-loss":0,
        "hours-per-week":70,
        "native-country":"United-States"
    },
    {
        "age":49,
        "workclass":"Self-emp-not-inc",
        "fnlgt":56841,
        "education":"Bachelors",
        "education-num":13,
        "marital-status":"Married-civ-spouse",
        "occupation":"Farming-fishing",
        "relationship":"Husband",
        "race":"White",
        "sex":"Male",
        "capital-gain":0,
        "capital-loss":0,
        "hours-per-week":70,
        "native-country":"United-States"
    }
]

Data Examples ("salary":">50K"):

[
    {
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
    },
    {
        "age":46,
        "workclass":"Private",
        "fnlgt":241935,
        "education":"11th",
        "education-num":7,
        "marital-status":"Married-civ-spouse",
        "occupation":"Other-service",
        "relationship":"Husband",
        "race":"Black",
        "sex":"Male",
        "capital-gain":7688,
        "capital-loss":0,
        "hours-per-week":40,
        "native-country":"United-States"
    }
]

"""

import requests

def main():

    # define the URL of the API endpoint
    url_local = 'http://127.0.0.1:8000/model/predict'
    url_cloud = 'http://127.0.0.1:8000/model/predict'

    # define the payload data to send as a dictionary
    payload = {
        "age":46,
        "workclass":"Private",
        "fnlgt":241935,
        "education":"11th",
        "education-num":7,
        "marital-status":"Married-civ-spouse",
        "occupation":"Other-service",
        "relationship":"Husband",
        "race":"Black",
        "sex":"Male",
        "capital-gain":7688,
        "capital-loss":0,
        "hours-per-week":40,
        "native-country":"United-States"
    }

    # send the POST request with the payload data
    response = requests.post(url, json=payload)

    return response

if __name__ == "__main__":
    # print the response from the server

    my_response = main()
    print(my_response.status_code)
    print(my_response.json())