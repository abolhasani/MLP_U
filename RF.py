import numpy as np
import pandas as pd
import random
import math
#from Metrics import *
from DecisionTree import ID3_rf, error, replace_none
from sklearn.metrics import accuracy_score, r2_score, f1_score, roc_auc_score, recall_score, precision_score

def preprocess_and_rearrange(data, label_column):
    label = data[label_column]
    data = data.drop(label_column, axis=1)
    data[label_column] = label
    return data

# my new preprocessing function for the diabetes dataset
def preprocess_diabetes_data(data, end_goal):
    for column in ['num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'service_utilization', 'number_of_medications', 'number_of_medication_changes']:
        median_val = data[column].median()
        data[column] = (data[column] > median_val).astype(int)
    columns_to_drop = ['diag_1', 'diag_2', 'diag_3']
    data.drop(columns=columns_to_drop, inplace=True)
    X = data.drop(end_goal, axis=1)
    y = data[end_goal]
    return X, y

def balanced_bootstrap_sampling(data, label_column, sample_size):
    unique_labels = data[label_column].unique()
    samples_per_label = sample_size // len(unique_labels)
    balanced_sample = pd.DataFrame()

    for label in unique_labels:
        label_sample = data[data[label_column] == label].sample(n=samples_per_label, replace=True)
        balanced_sample = pd.concat([balanced_sample, label_sample])

    return balanced_sample

def random_forest_data(train, test, end_label, max_trees, max_features=None, sample_size=None, heuristic='HS'):
    trees = []
    for n_trees in range(1, max_trees + 1):
        #bootstrap = train.sample(n=len(train), replace=True)
        bootstrap = balanced_bootstrap_sampling(train, end_label, sample_size)
        tree = ID3_rf(bootstrap.values, list(range(bootstrap.shape[1] - 1)), end_label, max_features, heuristic)
        trees.append(tree)
    print("Done making the trees!\n")
    return trees

def majority_vote(arr):
    arr = arr[~np.isnan(arr)]  
    arr = arr[arr >= 0]        
    if len(arr) == 0:
        return np.nan  # return NaN if no valid predictions
    return np.bincount(arr.astype(int)).argmax()

def calculate_metrics(trees, test_set):
    test_predictions = np.array([error(tree, test_set)[1] for tree in trees]).T
    test_predictions = np.where(test_predictions == None, np.nan, test_predictions).astype(float)
    majority_vote_test = np.apply_along_axis(majority_vote, axis=1, arr=test_predictions)
    valid_indices = ~np.isnan(majority_vote_test)
    majority_vote_test = majority_vote_test[valid_indices]
    y_true = test_set.iloc[:, -1].values[valid_indices]
    y_ord = test_set.iloc[:, -1].values
    errors = np.sum(majority_vote_test != y_true)
    error_rate = errors / len(y_true) if len(y_ord) > 0 else np.nan
    test_acc = accuracy_score(y_true, majority_vote_test) if len(y_true) > 0 else np.nan
    r_squared = r2_score(y_true, majority_vote_test) if len(y_true) > 0 else np.nan
    f1 = f1_score(y_true, majority_vote_test, average='weighted') if len(y_true) > 0 else np.nan
    auc = roc_auc_score(y_true, majority_vote_test) if (len(np.unique(y_true)) == 2 and len(y_true) > 0) else np.nan
    recall = recall_score(y_true, majority_vote_test, average='weighted') if len(y_true) > 0 else np.nan
    precision = precision_score(y_true, majority_vote_test, average='weighted') if len(y_true) > 0 else np.nan
    return 1-error_rate, test_acc, r_squared, f1, auc, recall, precision
