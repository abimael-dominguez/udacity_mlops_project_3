# from typing import Optional, List
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
        X,
        categorical_features: list = None,
        label=None,
        training=True,
        encoder=None,
        lb=None,
        mapping_labels=None):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=None)
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    mapping_labels(dict): A dictionary to explicitly set the positive value to a desired category
                          of the 'label' column.
                          Example: mapping_labels = {'>50K': 1, '<=50K': 0}
                          This dictionary is used to impute the labels also.
                          In this example '>50K' is explicitly set as the positive value.
                          Note: this parameter is only used when 'label' parameter is not None.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if categorical_features is None:
        categorical_features = []

    if label is not None:
        # inpute labels according to mapping_labels dictionary
        X[label] = X[label].map(mapping_labels)
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        # for sklearn 1.0.2; for newer versions use sparse_output=False
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        lb = LabelBinarizer(pos_label=1)
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            print(AttributeError)

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
