import pandas as pd
import numpy as np
from DecisionTree import *
#from sklearn.preprocessing import StandardScaler
from VMetrics import *

def load_and_prepare_data(file_path, target_column, test_size=0.3):
    data = pd.read_csv(file_path)

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=test_size)
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    return X_train, X_test, y_train.values.reshape(-1, 1), y_test.values.reshape(-1, 1)

def compute_cost(X, y, weights):
    m = len(y)
    predictions = X.dot(weights)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def stochastic_gradient_descent(X, y, learning_rate=0.01, max_iter=10000):
    m, n = X.shape
    weights = np.random.randn(n, 1) * 0.01  # Small random values
    for iteration in range(max_iter):
        i = np.random.randint(m)
        X_i = X[i:i+1]
        y_i = y[i:i+1]
        grad = gradient(X_i, y_i, weights)
        weights -= learning_rate * grad

        # Adaptive learning rate decay
        if iteration % 100 == 0 and iteration > 0:
            learning_rate *= 0.9

    return weights

def gradient(X, y, weights):
    m = len(y)
    predictions = X.dot(weights)
    grad = -(1 / m) * X.T.dot(y - predictions)
    return grad

def run_gradient_descent(file_path, target_column):
    X_train, X_test, y_train, y_test = load_and_prepare_data(file_path, target_column)

    weights = stochastic_gradient_descent(X_train, y_train)
    predictions = X_test.dot(weights)

    accuracy, r_squared, f1, auc, recall, precision = metricsSV(y_test, predictions)
    print("Accuracy: ", accuracy, "\nR2: ", r_squared, "\nF1: ", f1, "\nAUC: ", auc, "\nRecall: ", recall, "\nPrecision: ", precision)

# Example usage
print("For Readmitted Task: ")
run_gradient_descent('smote_re_sv.csv', 'readmitted')
print("For Time in Hospital Task: ")
run_gradient_descent('smote_time_sv.csv', 'time_in_hospital')


"""
def stochastic_gradient_descent(X, y, learning_rate=0.01, max_iterations=1000):
    m, n = X.shape
    weights = np.zeros((n, 1))
    costs = []

    for iteration in range(max_iterations):
        cost = 0
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            prediction = xi.dot(weights)
            gradient = xi.T.dot(prediction - yi)
            weights -= learning_rate * gradient
            cost += compute_cost(xi, yi, weights)
        
        avg_cost = cost / m
        costs.append(avg_cost)

        if iteration % 100 == 0:
            learning_rate /= 2

    return weights, costs, learning_rate

def run_gradient_descent(file_path, target_column):
    X_train, X_test, y_train, y_test = load_and_prepare_data(file_path, target_column)

    weights, costs, lr = stochastic_gradient_descent(X_train, y_train, learning_rate=0.01, max_iterations=1000)
    print("Weights: ", weights, "\nCost: ", costs[-1], "\nLearning Rate: ", lr)

    # Test set performance
    m = len(y_test)
    predictions = X_test.dot(weights)
    accuracy, r_squared, f1, auc, recall, precision = metricsSV(y_test, predictions)
    print("Accuracy: ", accuracy, "\nR2: ", r_squared, "\nF1: ", f1, "\nAUC: ", auc, "\nRecall: ", recall, "\nPrecision: ", precision)

# Example usage
run_gradient_descent('smote_re_sv.csv', 'readmitted')
run_gradient_descent('smote_time_sv.csv', 'time_in_hospital')
"""