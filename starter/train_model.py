# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_roc_curve
import joblib

# Add code to load in the data.
data = pd.read_csv('../data/clean_census.csv', encoding='utf-8')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
# Split the data prior to process

print("Process Data")

train, test = train_test_split(data, test_size=0.2, random_state=42)

mapping = {'>50K': 1, '<=50K': 0}
test['salary'] = test['salary'].map(mapping)
train['salary'] = train['salary'].map(mapping)

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

X_test, y_test, encoder_test, label_test = process_data(
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

# Calculate metrics
compute_roc_curve(X_test, y_test, best_model=best_nb)
compute_model_metrics(y=y_test, preds=best_nb.predict(X_test))
joblib.dump(best_nb, '../model/naive_bayes_model.pkl')

print("Finish training")



# Model inference
#model = joblib.load('../model/naive_bayes_model.pkl')
#inference(model=model, X=X_test)