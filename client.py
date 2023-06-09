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
    url_local = 'http://127.0.0.1:8000/model/predict'  # localhost
    # this will be unavailable soon
    url_cloud = 'https://render-project3-ml-devops-api.onrender.com/model/predict'

    # define the payload data to send as a dictionary
    payload = {
        "age": 46,
        "workclass": "Private",
        "fnlgt": 241935,
        "education": "11th",
        "education-num": 7,
        "marital-status": "Married-civ-spouse",
        "occupation": "Other-service",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 7688,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    print("url_local: ", url_local)
    print("url_cloud: ", url_cloud)

    # send the POST request with the payload data
    response = requests.post(url_cloud, json=payload)

    return response


if __name__ == "__main__":
    # print the response from the server

    my_response = main()
    print(my_response.status_code)
    print(my_response.json())

    server_header = my_response.headers.get('Server')
    if server_header:
        hostname = server_header.split('/')[0]
        print(f"Response came from server: {hostname}")
    else:
        print("Unable to determine server information from response headers.")

    source_url = my_response.request.url
    print(f"Response came from: {source_url}")
