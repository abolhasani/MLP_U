from DecisionTree import *
import pandas as pd

columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
train = pd.read_csv('Data/Bank/train.csv', names=columns)
train, train_with_labels = data_bank(train)
test = pd.read_csv('Data/Bank/test.csv', names=columns)
test, _ = data_bank(test,  train_with_labels)
max_trees = 500  # use 10 for testing, 500 for full run
n_iterations = 100  # use 10 for testing, 100 for full run
sample_size = 1000
# the bias_variance_decomposition function runs the required iterations on bagged trees (introduced from the last part of the problem - code in DecisionTree.py library)
avg_bias_single_tree, avg_variance_single_tree, avg_bias_bagging, avg_variance_bagging, avg_squared_error_single_tree, avg_squared_error_bagging = bias_variance_decomposition(train, test, n_iterations, max_trees, sample_size)
print(f"Average Bias (Single Tree): {avg_bias_single_tree}")
print(f"Average Variance (Single Tree): {avg_variance_single_tree}")
print(f"Average Squared Error (Single Tree): {avg_squared_error_single_tree}")
print(f"Average Bias (Bagging): {avg_bias_bagging}")
print(f"Average Variance (Bagging): {avg_variance_bagging}")
print(f"Average Squared Error (Bagging): {avg_squared_error_bagging}")
