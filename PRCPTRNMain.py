import pandas as pd
import numpy as np
from perceptron import *
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

def load_data(filepath, target_column, test_size=0.3):
    # Load the dataset
    df = pd.read_csv(filepath)

    # Separate features and target
    features = df.drop(columns=[target_column]).values
    labels = df[target_column].values

    # Convert labels to {-1, 1} to match the sign function
    labels = np.where(labels == 0, -1, 1)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = custom_train_test_split(features, labels, test_size=test_size)

    return X_train, X_test, y_train, y_test


def stdperc():
    file_path = 'smote_re_sv.csv'
    target_column = 'readmitted'
    train_features, test_features, train_labels, test_labels = load_data(file_path, target_column, test_size=0.3)
    p = Perceptron(num_features=train_features.shape[1])
    p.train(train_features, train_labels)
    test_predictions = np.array([p.predict(x) for x in test_features])
    test_predictions_converted = (test_predictions + 1) // 2
    test_labels_converted = (test_labels + 1) // 2
    #accuracy, r_squared, f1, auc, recall, precision = metricsSV(test_labels, test_predictions)
    accuracy, r_squared, f1, auc, recall, precision = metricsSV(test_labels_converted, test_predictions_converted)
    print(f'Standard Perceptron Metrics:')
    print(f'Accuracy: {accuracy}, R2: {r_squared}, F1: {f1}, AUC: {auc}, Recall: {recall}, Precision: {precision}')

def votedperc():
    file_path = 'smote_re_sv.csv'
    target_column = 'readmitted'
    train_features, test_features, train_labels, test_labels = load_data(file_path, target_column, test_size=0.3)
    voted_perceptron = VotedPerceptron(num_features=train_features.shape[1])
    voted_perceptron.train(train_features, train_labels, T=5)
    test_predictions = voted_perceptron.predict(test_features)
    test_predictions_converted = (test_predictions + 1) // 2
    test_labels_converted = (test_labels + 1) // 2
    #accuracy, r_squared, f1, auc, recall, precision = metricsSV(test_labels, test_predictions)
    accuracy, r_squared, f1, auc, recall, precision = metricsSV(test_labels_converted, test_predictions_converted)
    print(f'Voted Perceptron Metrics:')
    print(f'Accuracy: {accuracy}, R2: {r_squared}, F1: {f1}, AUC: {auc}, Recall: {recall}, Precision: {precision}')

def aveperc():
    file_path = 'smote_re_sv.csv'
    target_column = 'readmitted'
    train_features, test_features, train_labels, test_labels = load_data(file_path, target_column, test_size=0.3)
    avg_perceptron = AveragedPerceptron(num_features=train_features.shape[1])
    avg_perceptron.train(train_features, train_labels, T=5)
    test_predictions = avg_perceptron.predict(test_features)
    test_predictions_converted = (test_predictions + 1) // 2
    test_labels_converted = (test_labels + 1) // 2
    #accuracy, r_squared, f1, auc, recall, precision = metricsSV(test_labels, test_predictions)
    accuracy, r_squared, f1, auc, recall, precision = metricsSV(test_labels_converted, test_predictions_converted)
    print(f'Averaged Perceptron Metrics:')
    print(f'Accuracy: {accuracy}, R2: {r_squared}, F1: {f1}, AUC: {auc}, Recall: {recall}, Precision: {precision}')

def main():
    print("For Readmitted Task: ")
    print("Standard Perceptron: ")
    stdperc()
    print("Average Perceptron: ")
    aveperc()
    print("Voted Perceptron: ")
    votedperc()

if __name__ == '__main__':
    main()
