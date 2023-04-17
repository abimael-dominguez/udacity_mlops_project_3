"""

Sample of data:

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
        "native-country":"United-States",
        "salary":0
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
        "native-country":"United-States",
        "salary":0
    },
    {
        "age":55,
        "workclass":"Private",
        "fnlgt":125000,
        "education":"Masters",
        "education-num":14,
        "marital-status":"Divorced",
        "occupation":"Exec-managerial",
        "relationship":"Unmarried",
        "race":"White",
        "sex":"Male",
        "capital-gain":0,
        "capital-loss":0,
        "hours-per-week":40,
        "native-country":"United-States",
        "salary":1
    },
    {
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
        "native-country":"United-States",
        "salary":1
    }
]


"""

# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_roc_curve, compute_metrics_on_slices
import joblib

# Add code to load in the data.
data = pd.read_csv('../data/clean_census.csv', encoding='utf-8')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
# Split the data prior to process

print("Process Data")

train, test = train_test_split(data, test_size=0.2, random_state=42)


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

X_train, y_train, encoder, label = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process the test data with the process_data function.

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=label,
)

print("Finish Process Data")
print("Start training")

# Train and save a model.
best_nb = train_model(X_train, y_train)

# Calculate overall metrics
predictions_on_test = best_nb.predict(X_test)
compute_roc_curve(X_test, y_test, best_model=best_nb)
precision, recall, fbeta = compute_model_metrics(y=y_test, preds=predictions_on_test)
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"fbeta: {fbeta}")
joblib.dump(best_nb, '../model/naive_bayes_model.pkl')
joblib.dump(encoder, '../model/encoder.pkl')

print("Finish training")

print("Calculating Performance on Slices ...")

slices_data = test
slices_data['predictions'] = predictions_on_test
slices_results = []

for cat_feat in cat_features:
    temp_results = compute_metrics_on_slices(
        data=slices_data,
        slice_column=cat_feat,
        label_column='salary',
        prediction_column='predictions')
    
    slices_results += temp_results

slices_df = round(pd.DataFrame(slices_results), 4)
slices_df = slices_df.sort_values(by=['recall', 'precision', 'fbeta'])  # ascending=False
slices_df.to_csv('../data/slice_output.txt', encoding='utf-8', sep=',', index=False)
slices_df.to_csv('../data/slice_output.csv', encoding='utf-8', sep=',', index=False)

print("Finish train_model.py")

# Model inference
#model = joblib.load('../model/naive_bayes_model.pkl')
#inference(model=model, X=X_test)