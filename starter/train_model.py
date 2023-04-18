"""
Train the model
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

mask = (test['education'] == '11th') & (test['salary'] == '>50K')
print((test[mask]).head(2).to_dict(orient='records'))

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

X_train, y_train, encoder, label_binarizer = process_data(
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
    lb=label_binarizer,
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

# Save the model 
joblib.dump(best_nb, '../model/naive_bayes_model.pkl')
joblib.dump(encoder, '../model/encoder.pkl')
joblib.dump(label_binarizer, '../model/label_binarizer.pkl')

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