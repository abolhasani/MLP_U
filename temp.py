import numpy as np
import pandas as pd
import random
import math
#from Metrics import *
from DecisionTree import ID3_rf, error, replace_none, custom_train_test_split
from sklearn.metrics import accuracy_score, r2_score, f1_score, roc_auc_score, recall_score, precision_score


def preprocess_and_rearrange(data, label_column):
    # Move the label column to the end of the DataFrame
    label = data[label_column]
    data = data.drop(label_column, axis=1)
    data[label_column] = label
    return data

# New preprocessing function for the diabetes dataset
def preprocess_diabetes_data(data, end_goal):
    for column in ['num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'service_utilization', 'number_of_medications', 'number_of_medication_changes']:
        median_val = data[column].median()
        data[column] = (data[column] > median_val).astype(int)
    # Split data into features and label
    #columns_to_drop = ['diag_1', 'diag_2', 'diag_3']
    #data.drop(columns=columns_to_drop, inplace=True)
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
    #train_errors = []
    #test_errors = []

    for n_trees in range(1, max_trees + 1):
        #print(n_trees)
        #bootstrap = train.sample(n=len(train), replace=True)
        bootstrap = balanced_bootstrap_sampling(train, end_label, sample_size)
        tree = ID3_rf(bootstrap.values, list(range(bootstrap.shape[1] - 1)), end_label, max_features, heuristic)
        trees.append(tree)
        # Error calculation omitted for brevity, can be included as before
    print("Done making the trees!\n")
    return trees

def calculate_metrics(trees, test_set):
    # Generating predictions for each tree
    test_predictions = np.array([error(tree, test_set)[1] for tree in trees]).T

    # Convert None to np.nan and ensure numeric types
    test_predictions = np.where(test_predictions == None, np.nan, test_predictions).astype(float)

    # Function to perform majority vote while excluding -1 and NaN values
    def majority_vote(arr):
        arr = arr[~np.isnan(arr)]  # Exclude NaN values
        arr = arr[arr >= 0]        # Exclude negative values
        if len(arr) == 0:
            return np.nan  # Return NaN if no valid predictions
        return np.bincount(arr.astype(int)).argmax()

    # Calculating majority vote
    majority_vote_test = np.apply_along_axis(majority_vote, axis=1, arr=test_predictions)

    # Exclude rows where majority vote resulted in NaN (i.e., no valid predictions)
    valid_indices = ~np.isnan(majority_vote_test)
    majority_vote_test = majority_vote_test[valid_indices]
    y_true = test_set.iloc[:, -1].values[valid_indices]
    y_ord = test_set.iloc[:, -1].values

    # Compute error rate
    errors = np.sum(majority_vote_test != y_true)
    error_rate = errors / len(y_true) if len(y_ord) > 0 else np.nan

    # Metrics calculation
    test_acc = accuracy_score(y_true, majority_vote_test) if len(y_true) > 0 else np.nan
    r_squared = r2_score(y_true, majority_vote_test) if len(y_true) > 0 else np.nan
    f1 = f1_score(y_true, majority_vote_test, average='weighted') if len(y_true) > 0 else np.nan
    auc = roc_auc_score(y_true, majority_vote_test) if (len(np.unique(y_true)) == 2 and len(y_true) > 0) else np.nan
    recall = recall_score(y_true, majority_vote_test, average='weighted') if len(y_true) > 0 else np.nan
    precision = precision_score(y_true, majority_vote_test, average='weighted') if len(y_true) > 0 else np.nan

    return 1-error_rate, test_acc, r_squared, f1, auc, recall, precision



data = pd.read_csv('smote_re.csv') # Load your dataset
# Preprocess your data
data = preprocess_and_rearrange(data, 'readmitted')
X, y = preprocess_diabetes_data(data, 'readmitted')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.1)

# Combine features and labels for train and test sets
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# Set your parameters
subset_feature_size = 3 # Example value
number_of_trees = 1000 # Example value
sample_size = 2000 # Example value

number_of_trees = 500  # Equivalent to n_estimators in scikit-learn
#subset_feature_size = int(math.sqrt(X_train.shape[1]))  # Default scikit-learn behavior for classification
sample_size = len(train)  # Use the entire dataset for each bootstrap sample

# Run Random Forest
trees = random_forest_data(train, test, 'readmitted', number_of_trees, subset_feature_size, sample_size, 'ME')

# Calculate Metrics
accu, test_acc, r_squared, f1, auc, recall, precision = calculate_metrics(trees, test)
print(f"Accuracy: {accu}, Test Accuracy: {test_acc}, R-squared: {r_squared}, F-1 Score: {f1}, AUC: {auc}, Recall: {recall}, Precision: {precision}")


