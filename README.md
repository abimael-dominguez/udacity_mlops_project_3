# Machine Learning DevOps Engineer - Project 3

--------------------------------

# Data Set Information:

Extraction was done by Barry Becker from the 1994 Census database.
A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

Prediction task is to determine whether a person makes over 50K a year.


| Column           | Description                                                                                                                   |
|------------------|-------------------------------------------------------------------------------------------------------------------------------|
| Salary           | >50K, <=50K                                                                                                                   |
| age              | continuous                                                                                                                     |
| workclass        | Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked                         |
| fnlwgt           | continuous                                                                                                                     |
| education        | Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, <br>7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool |
| education-num    | continuous                                                                                                                     |
| marital-status   | Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse                      |
| occupation       | Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,<br> Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, <br> Protective-serv, Armed-Forces |
| relationship     | Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried                                                             |
| race             | White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black                                                                     |
| sex              | Female, Male                                                                                                                   |
| capital-gain     | continuous                                                                                                                     |
| capital-loss     | continuous                                                                                                                     |
| hours-per-week   | continuous                                                                                                                     |
| native-country   | United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece,<br> South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France,<br>  Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, <br>Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands. |

# Training
If you want to train or retrain the model you should go to the directory: ./starter.
Don't run the script from the outside this directory.
- command:
    - pyhton train_model.py

# Testing (pytest)
- In tha main directory run the following:
    - pytest -vv 

# Run the API
- To run the API:
    - uvicorn main:app --reload
- To run the client:
    - python client.py

Note: the client.py can make request to th local API o to the API in the cloud, please adapt the code so you query the service which is available. 

# GitHub link
    - https://github.com/abimael-dominguez/udacity_mlops_project_3

# Apply Autopep8
- To clean the code the code and assure CI phase succeed you can apply:
    - autopep8 --in-place --aggressive --aggressive client.py conftest.py main.py test_main.py test_ml.py starter/ml/model.py starter/ml/data.py starter/train_model.py

# CI with GitHub Actions (flake8)
- If it encounters any errors, it causes the CI process to stop and report a failure. The GitHub editor is 127 chars wide.
    - flake8 . --count --max-complexity=18 --max-line-length=127 --statistics

# CD in Render Cloud
The GitHub repository was connected to Render. Automatic deployment was enabled for every "push" send to the repository (If CI succeed)


# More information
For more information about this model please see the model_card.md.

# Project instructions

## Environment Set up

Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.8" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.

## Repositories
* Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.

## Data
* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

## Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

## API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
    * Type hinting must be used.
    * Use a Pydantic model to ingest the body from POST. This model should contain an example.
   	 * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

## API Deployment
* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
    * Enable automatic deployments that only deploy if your continuous integration passes.
    * Hint: think about how paths will differ in your local environment vs. on Heroku.
    * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Write a script that uses the requests module to do one POST on your live API.