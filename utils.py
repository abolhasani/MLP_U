import numpy as np
import random

def custom_train_test_split(X, y, test_size=0.3):
    m = len(y)
    test_size = int(m * test_size)
    indices = np.random.permutation(m)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

# Function to manually split the dataset into k folds
def manual_k_fold_split(X, y, k=5):
    fold_size = len(X) // k
    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        X_train = np.concatenate([X[:start], X[end:]])
        y_train = np.concatenate([y[:start], y[end:]])
        X_test, y_test = X[start:end], y[start:end]
        yield X_train, X_test, y_train, y_test