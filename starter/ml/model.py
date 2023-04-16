from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Define the parameter grid to search over
    #param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]}

    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
        'priors': [None, [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2]],
    }

    # Create a Gaussian Naive Bayes model
    nb = GaussianNB()

    # Use GridSearchCV to search over the parameter grid
    grid_search = GridSearchCV(nb, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Return the best model
    return grid_search.best_estimator_



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """

    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def compute_roc_curve(X_test, y_test, best_model):
    """
    Create a picture of the ROC curve.
    This function was tested with a Naive Bayes model from Scikit Learn.
    """
    # Evaluate the model on the test data
    probs = best_model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probs[:,1], pos_label=1)
    auc = roc_auc_score(y_test, probs[:,1])

    print(f'Best hyperparameters: {best_model.get_params()}')
    print(f'Test AUC: {auc}')

    # Plot the ROC curve
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (AUC = {:.2f})'.format(auc))

    # Save the plot to an image file
    plt.savefig('../screenshots/roc_curve.png')

def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    

    # Make predictions using the loaded model
    #X_new = [[1.0, 2.0, 3.0, 4.0]]
    y_pred = model.predict(X)
