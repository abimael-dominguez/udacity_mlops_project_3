"""
Main API.

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


- Fast API references:
    - https://fastapi.tiangolo.com/tutorial/first-steps/
    - https://fastapi.tiangolo.com/tutorial/schema-extra-example/#__tabbed_1_2

"""
from fastapi import FastAPI, Body
# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field
from typing_extensions import Annotated  # for python 3.9 use from typing import Annotated
import pandas as pd
import joblib

from starter.ml.data import process_data
from starter.ml.model import inference

# Declare the data object with its components and their type.
class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list]
    item_id: int

# Save items from POST method in the memory
items = {}

# Initialize FastAPI instance
app = FastAPI()

# Welcome message in the root
@app.get("/")
def read_root():
    return {"greeting": "Hello World!"}

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/items/")
async def create_item(item: TaggedItem):
    items[item.item_id] = item
    return item


# A GET that in this case just returns the item_id we pass,
# but a future iteration may link the item_id here to the one we defined in our TaggedItem.
@app.get("/items/{item_id}")
async def get_items(item_id: int, count: int = 1):
    try:
        item = items[item_id]
    except:
        return "Item not found."

    return {"fetch": f"Fetched: {item.name} with qty of {count}"}

# --------------- INFERENCE ---------------

class Person(BaseModel):
    age: int
    workclass: str
    fnlgt: Union[int, float]
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: Union[int, float] = Field(..., alias="capital-gain")
    capital_loss: Union[int, float] = Field(..., alias="capital-loss")
    hours_per_week: Union[int, float] = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

example_person = Annotated[
                            Person,
                            Body(
                                example={
                                    "age":57,
                                    "workclass":"Federal-gov",
                                    "fnlgt":414994,
                                    "education":"Some-college",
                                    "education-num":10,
                                    "marital-status":"Married-civ-spouse",
                                    "occupation":"Exec-managerial",
                                    "relationship":"Husband",
                                    "race":"White",
                                    "sex":"Male",
                                    "capital-gain":0,
                                    "capital-loss":0,
                                    "hours-per-week":40,
                                    "native-country":"United-States"
                                }
                            )
                        ]

# Inference
@app.post("/model/predict/")
async def get_predictions(person: example_person):


    data = [
        {
            'age': person.age,
            'workclass': person.workclass,
            'fnlgt': person.fnlgt,
            'education': person.education,
            'education-num': person.education_num,
            'marital-status': person.marital_status,
            'occupation': person.occupation,
            'relationship': person.relationship,
            'race': person.race,
            'sex': person.sex,
            'capital-gain': person.capital_gain,
            'capital-loss': person.capital_loss,
            'hours-per-week': person.hours_per_week,
            'native-country': person.native_country
        }
    ]

    df = pd.DataFrame(data)

    # Be sure of the column order
    df.columns = ["age"
                  ,"workclass"
                  ,"fnlgt"
                  ,"education"
                  ,"education-num"
                  ,"marital-status"
                  ,"occupation"
                  ,"relationship"
                  ,"race"
                  ,"sex"
                  ,"capital-gain"
                  ,"capital-loss"
                  ,"hours-per-week"
                  ,"native-country"]

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    
    # Load model
    
    model = joblib.load('./model/naive_bayes_model.pkl')
    encoder = joblib.load('./model/encoder.pkl')
    label_binarizer = joblib.load('./model/label_binarizer.pkl')
    
    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=label_binarizer,
    )

    my_prediction = inference(model=model, X=X)

    result = {
        "query":data[0],
        "prediction": int(my_prediction[0])
    }

    return result