import numpy as np
import pandas as pd
from VMetrics import metricsSV

def custom_train_test_split(X, y, test_size=0.3):
    # Calculate the number of test samples
    m = len(y)
    test_size = int(m * test_size)

    # Generate random indices
    indices = np.random.permutation(m)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    # Split the data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights))
    cost = (-1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = sigmoid(np.dot(X, weights))
        weights -= (learning_rate/m) * np.dot(X.T, (predictions - y))
        cost_history.append(compute_cost(X, y, weights))

    return weights, cost_history

def predict(X, weights):
    predictions = sigmoid(np.dot(X, weights))
    return np.where(predictions >= 0.5, 1, 0)

def logistic_regression(X_train, y_train, X_test, y_test, learning_rate=0.01, iterations=1000):
    # Add intercept term to X_train and X_test
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    # Initialize weights
    weights = np.zeros(X_train.shape[1])

    # Perform gradient descent
    weights, _ = gradient_descent(X_train, y_train, weights, learning_rate, iterations)

    # Predicting labels and probabilities
    train_predictions = predict(X_train, weights)
    test_predictions = predict(X_test, weights)
    test_probabilities = sigmoid(np.dot(X_test, weights))

    # Calculate metrics including AUC
    #train_accuracy, _, train_f1, train_auc, train_recall, train_precision = metricsSV(y_train, train_predictions)
    test_accuracy, r_square, test_f1, test_auc, test_recall, test_precision = metricsSV(y_test, test_predictions)

    #print("Training Metrics:")
    #print(f"Accuracy: {train_accuracy}, F1: {train_f1}, AUC: {train_auc}, Recall: {train_recall}, Precision: {train_precision}")
    
    print("Testing Metrics:")
    print(f"Accuracy: {test_accuracy}, R2:{r_square}, F1: {test_f1}, AUC: {test_auc}, Recall: {test_recall}, Precision: {test_precision}")

    return weights

# Loading data and training logistic regression model
def run_logistic_regression(file_path, target_column):
    data = pd.read_csv(file_path)
    X = data.drop(columns=[target_column]).values
    y = data[target_column].values

    # Convert labels to 0 and 1
    y = np.where(y == -1, 0, y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.3)

    # Train and evaluate the model
    logistic_regression(X_train, y_train, X_test, y_test)

# Example usage
print("For Readmitted Task: ")
run_logistic_regression('smote_re_sv.csv', 'readmitted')
#print("For Time in Hospital Task: ")
#run_logistic_regression('smote_time_sv.csv', 'time_in_hospital')
