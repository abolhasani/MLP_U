import pandas as pd
import numpy as np
from DecisionTree import *
from sklearn.preprocessing import StandardScaler
from VMetrics import *

def load_and_prepare_data(file_path, target_column, test_size=0.3):
    data = pd.read_csv(file_path)

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=test_size)

    # Standardize features
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    # Adding a bias term to the data
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    return X_train, X_test, y_train.values.reshape(-1, 1), y_test.values.reshape(-1, 1)

def compute_cost(X, y, weights):
    m = len(y)
    predictions = X.dot(weights)
    cost = (1/(2*m)) * np.sum(np.square(predictions-y))
    #cost = (1/(2)) * np.sum(np.square(predictions-y))
    return cost

def batch_gradient_descent(X, y, learning_rate=0.05, tolerance=1e-6, max_iterations=1000):
    # ... [Existing batch_gradient_descent function code] ...
    m, n = X.shape
    weights = np.zeros((n, 1))
    costs = []
    for iteration in range(max_iterations):
        gradient = (1/m) * X.T.dot(X.dot(weights) - y)
        prev_weights = weights.copy()
        weights -= learning_rate * gradient
        if iteration % 100 == 0:
            learning_rate /= 2
        # Convergence check
        weight_difference_norm = np.linalg.norm(weights - prev_weights)
        if weight_difference_norm < tolerance:
            print("CONVERGED!")
            break
        cost_i=compute_cost(X, y, weights)
        costs.append(cost_i)
    return weights, costs, learning_rate

"""
def run_gradient_descent(file_path, target_column):
    X_train, X_test, y_train, y_test = load_and_prepare_data(file_path, target_column)

    weights, costs, lr = batch_gradient_descent(X_train, y_train, learning_rate=0.5)
    #print("Weights: ", weights, "\nCost: ", costs[-1], "\nLearning Rate: ", lr)
    # Test set performance
    m = len(y_test)
    predictions = X_test.dot(weights)
    #test_cost = (1/(2 * m)) * np.sum(np.square(predictions - y_test))
    if (target_column=='readmitted'):
        accuracy, r_squared, f1, auc, recall, precision = metricsSV(y_test, predictions)
    else:
        accuracy, r_squared, f1, auc, recall, precision = metricsSV(y_test, predictions)
    print("Accuracy: ", accuracy, "\nR2: ", r_squared, "\nF1: ", f1, "\nAUC: ", auc, "\nRecall: ", recall, "\nPrecision: ", precision)
    #print("Cost of the test data with the latest weight: ", test_cost)
"""
def run_gradient_descent(file_path, target_column):
    X_train, X_test, y_train, y_test = load_and_prepare_data(file_path, target_column)

    # Optional: Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    weights, costs, lr = batch_gradient_descent(X_train, y_train, learning_rate=0.5)

    # Convert predictions to binary labels (0 or 1)
    predictions = np.dot(X_test, weights)
    predictions_binary = np.where(predictions >= 0.5, 1, 0)

    # Calculate metrics
    accuracy, r_squared, f1, auc, recall, precision = metricsSV(y_test, predictions_binary)
    print("Metrics:")
    print("Accuracy:", accuracy, "R2:", r_squared, "F1:", f1, "AUC:", auc, "Recall:", recall, "Precision:", precision)


# Example usage
run_gradient_descent('smote_re_sv.csv', 'readmitted')
run_gradient_descent('smote_time_sv.csv', 'time_in_hospital')
