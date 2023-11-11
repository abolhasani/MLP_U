from sklearn.metrics import f1_score, r2_score, recall_score, precision_score, roc_auc_score, accuracy_score
import numpy as np
from scipy.special import softmax

def metricsSV(y_true, y_pred):
    # Convert y_true to a 1D array if it's a 2D array of shape (n, 1)
    if y_true.ndim > 1 and y_true.shape[1] == 1:
        y_true = y_true.ravel()

    # Check if the classification is binary or multiclass
    unique_classes = np.unique(y_true)
    binary = len(unique_classes) == 2

    # Handle binary and multiclass predictions
    if binary:
        # Convert y_pred to binary if binary classification
        y_pred_binary = (y_pred > 0.5).astype(int)
        auc = roc_auc_score(y_true, y_pred_binary)
    else:
        # Convert y_pred to class labels for multiclass classification
        y_pred_binary = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else np.rint(y_pred).astype(int)
        auc = None  # AUC is not applicable for multiclass

    accuracy = accuracy_score(y_true, y_pred_binary)
    r_squared = r2_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_binary, average='macro')
    recall = recall_score(y_true, y_pred_binary, average='macro')
    precision = precision_score(y_true, y_pred_binary, average='macro')

    return accuracy, r_squared, f1, auc, recall, precision


"""
def metricsSV(y_true, y_pred):
    # Convert y_true to a 1D array if it's a 2D array of shape (n, 1)
    if y_true.ndim > 1 and y_true.shape[1] == 1:
        y_true = y_true.ravel()

    # Check if the classification is binary or multiclass
    unique_classes = np.unique(y_true)
    binary = len(unique_classes) == 2

    # Convert y_pred to binary if binary classification
    y_pred_binary = (y_pred > 0.5).astype(int) if binary else y_pred

    accuracy = accuracy_score(y_true, y_pred_binary)
    r_squared = r2_score(y_true, y_pred)

    # Select averaging method based on binary or multiclass
    average_method = 'binary' if binary else 'macro'
    f1 = f1_score(y_true, y_pred_binary, average=average_method)
    recall = recall_score(y_true, y_pred_binary, average=average_method)
    precision = precision_score(y_true, y_pred_binary, average=average_method)

    auc = roc_auc_score(y_true, y_pred_binary) if binary else None

    return accuracy, r_squared, f1, auc, recall, precision
"""


# Usage Example
# predictions = X_test.dot(weights)
# accuracy, r_squared, f1, auc, recall, precision = metricsSV(y_test, predictions, binary=True)
# print("Accuracy: ", accuracy, "\nR2: ", r_squared, "\nF1: ", f1, "\nAUC: ", auc, "\nRecall: ", recall, "\nPrecision: ", precision)
