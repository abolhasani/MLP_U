from DecisionTree import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# initializing
columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
# importing data and preprocessing
train = pd.read_csv('Data/Bank/train.csv', names=columns)
train, train_with_labels = data_bank(train)
test = pd.read_csv('Data/Bank/test.csv', names=columns)
test, _ = data_bank(test,  train_with_labels)
# initializing depth and heuristic methods, and also our final results table data frame
max_trees = 500 # use 10 for test and 500 for full run
train_errors, test_errors, trees = bagged_trees(train, test, max_trees)
# plotting
plt.figure(figsize=(15, 10))
plt.plot(range(1, max_trees+1), train_errors, label="Train Error")
plt.plot(range(1, max_trees+1), test_errors, label="Test Error")
plt.xlabel("Number of Trees")
plt.ylabel("Error")
plt.legend()
plt.title("Bagged Trees Results' Figure")
plt.show()