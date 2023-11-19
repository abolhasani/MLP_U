from sklearn.metrics import f1_score, r2_score, recall_score, precision_score, roc_auc_score, accuracy_score
import numpy as np
from scipy.special import softmax

def metricsSV(y_true, y_pred):
    # convert y_true to a 1D array if it's a 2D array of shape (n, 1)
    if y_true.ndim > 1 and y_true.shape[1] == 1:
        y_true = y_true.ravel()
    unique_classes = np.unique(y_true)
    binary = len(unique_classes) == 2
    if binary:
        y_pred_binary = (y_pred > 0.5).astype(int)
        auc = roc_auc_score(y_true, y_pred_binary)
    else:
        y_pred_binary = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else np.rint(y_pred).astype(int)
        auc = None  
    accuracy = accuracy_score(y_true, y_pred_binary)
    r_squared = r2_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_binary, average='macro')
    recall = recall_score(y_true, y_pred_binary, average='macro')
    precision = precision_score(y_true, y_pred_binary, average='macro')
    return accuracy, r_squared, f1, auc, recall, precision
