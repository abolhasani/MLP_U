import numpy as np
import pandas as pd
from RF import *
from utils import manual_k_fold_split

# main model no k-fold cross validation for readmission
def custom_train_test_split1(X, y, test_size=0.2, random_state=None):
    # setting the random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    return X_train, X_test, y_train, y_test

data = pd.read_csv('smote_re.csv') 
data = preprocess_and_rearrange(data, 'readmitted')
X, y = preprocess_diabetes_data(data, 'readmitted')

X_train, X_test, y_train, y_test = custom_train_test_split1(X, y, test_size=0.1)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)
subset_feature_size = 3 
number_of_trees = 1000 
sample_size = 2000 
#subset_feature_size = int(math.sqrt(X_train.shape[1]))
number_of_trees = 100  
sample_size = len(train) 

trees = random_forest_data(train, test, 'readmitted', number_of_trees, subset_feature_size, sample_size, 'ME')
accu, test_acc, r_squared, f1, auc, recall, precision = calculate_metrics(trees, test)
print(f"Accuracy: {accu}, Test Accuracy: {test_acc}, R-squared: {r_squared}, F-1 Score: {f1}, AUC: {auc}, Recall: {recall}, Precision: {precision}")
# Accuracy: 0.7836817653890824, Test Accuracy: 0.7836817653890824, R-squared: 0.1345014872536352, F-1 Score: 0.7836842791208027, AUC: 0.7836375626934728, Recall: 0.7836817653890824, Precision: 0.7836876862151123

# main model no k-fold cross validation for stay time in hospital
data = pd.read_csv('smote_time.csv') 
data = preprocess_and_rearrange(data, 'time_in_hospital')
X, y = preprocess_diabetes_data(data, 'time_in_hospital')
X_train, X_test, y_train, y_test = custom_train_test_split1(X, y, test_size=0.1)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)
subset_feature_size = 3 
number_of_trees = 1000 
sample_size = 2000 
number_of_trees = 100  # Equivalent to n_estimators in scikit-learn
#subset_feature_size = int(math.sqrt(X_train.shape[1]))  
sample_size = len(train)  

trees = random_forest_data(train, test, 'time_in_hospital', number_of_trees, subset_feature_size, sample_size, 'ME')
accu, test_acc, r_squared, f1, auc, recall, precision = calculate_metrics(trees, test)
print(f"Accuracy: {accu}, Test Accuracy: {test_acc}, R-squared: {r_squared}, F-1 Score: {f1}, AUC: {auc}, Recall: {recall}, Precision: {precision}")
# Accuracy: 0.5496665247623103, Test Accuracy: 0.5496665247623101, R-squared: -0.23363681322669416, F-1 Score: 0.44598335493343394, AUC: nan, Recall: 0.5496665247623101, Precision: 0.382020085728277
"""


"""

print("\n5-fold cross validation for readmitted")
data = pd.read_csv('smote_re.csv')
data = preprocess_and_rearrange(data, 'readmitted')
X, y = preprocess_diabetes_data(data, 'readmitted')
number_of_trees_values = [50, 100]
subset_feature_size_values = [2, 3, 4, 6]  
sample_size_values = [500, 1000, 2000]  
results = []

# 5-Fold Cross-Validation
for number_of_trees in number_of_trees_values:
    for subset_feature_size in subset_feature_size_values:
        for sample_size in sample_size_values:
            print(f"num trees: {number_of_trees}, sub feature size: {subset_feature_size}, sample size: {sample_size}")
            accuracies = []
            for X_train, X_test, y_train, y_test in manual_k_fold_split(X, y):
                X_train_df = pd.DataFrame(X_train)
                y_train_df = pd.DataFrame(y_train, columns=['readmitted'])
                X_test_df = pd.DataFrame(X_test)
                y_test_df = pd.DataFrame(y_test, columns=['readmitted'])
                train = pd.concat([X_train_df, y_train_df], axis=1)
                test = pd.concat([X_test_df, y_test_df], axis=1)
                trees = random_forest_data(train, test, 'readmitted', number_of_trees, subset_feature_size, sample_size, 'ME')
                accu, test_acc, _, _, _, _, _ = calculate_metrics(trees, test)
                accuracies.append(accu)
            avg_accuracy = sum(accuracies) / len(accuracies)
            results.append((number_of_trees, subset_feature_size, sample_size, avg_accuracy))

best_params = max(results, key=lambda x: x[3])
train = pd.concat([X, y], axis=1)
trees = random_forest_data(train, train, 'readmitted', best_params[0], best_params[1], best_params[2], 'ME')
accu, test_acc, r_squared, f1, auc, recall, precision = calculate_metrics(trees, train)
print(f"Best Hyperparameters: Number of Trees: {best_params[0]}, Subset Feature Size: {best_params[1]}, Sample Size: {best_params[2]}")
print(f"Full Model Metrics - Accuracy: {accu}, Test Accuracy: {test_acc}, R-squared: {r_squared}, F-1 Score: {f1}, AUC: {auc}, Recall: {recall}, Precision: {precision}")
# Best Hyperparameters: Number of Trees: 100, Subset Feature Size: 4, Sample Size: 2000
# Full Model Metrics - Accuracy: 0.7594539035097907, Test Accuracy: 0.7594539035097907, R-squared: 0.03781561403916289, F-1 Score: 0.7591875571170539, AUC: 0.7594539035097907, Recall: 0.7594539035097907, Precision: 0.7606068621664038

print("\n5-fold cross validation for time_in_hospital")
data = pd.read_csv('smote_time.csv')
data = preprocess_and_rearrange(data, 'time_in_hospital')
X, y = preprocess_diabetes_data(data, 'time_in_hospital')
results = []

# 5-Fold Cross-Validation
for number_of_trees in number_of_trees_values:
    for subset_feature_size in subset_feature_size_values:
        for sample_size in sample_size_values:
            print(f"num trees: {number_of_trees}, sub feature size: {subset_feature_size}, sample size: {sample_size}")
            accuracies = []
            for X_train, X_test, y_train, y_test in manual_k_fold_split(X, y):
                X_train_df = pd.DataFrame(X_train)
                y_train_df = pd.DataFrame(y_train, columns=['time_in_hospital'])
                X_test_df = pd.DataFrame(X_test)
                y_test_df = pd.DataFrame(y_test, columns=['time_in_hospital'])
                train = pd.concat([X_train_df, y_train_df], axis=1)
                test = pd.concat([X_test_df, y_test_df], axis=1)
                trees = random_forest_data(train, test, 'time_in_hospital', number_of_trees, subset_feature_size, sample_size, 'ME')
                accu, test_acc, _, _, _, _, _ = calculate_metrics(trees, test)
                accuracies.append(accu)
            avg_accuracy = sum(accuracies) / len(accuracies)
            results.append((number_of_trees, subset_feature_size, sample_size, avg_accuracy))

best_params = max(results, key=lambda x: x[3])
train = pd.concat([X, y], axis=1)
trees = random_forest_data(train, train, 'time_in_hospital', best_params[0], best_params[1], best_params[2], 'ME')
accu, test_acc, r_squared, f1, auc, recall, precision = calculate_metrics(trees, train)
print(f"Best Hyperparameters: Number of Trees: {best_params[0]}, Subset Feature Size: {best_params[1]}, Sample Size: {best_params[2]}")
print(f"Full Model Metrics - Accuracy: {accu}, Test Accuracy: {test_acc}, R-squared: {r_squared}, F-1 Score: {f1}, AUC: {auc}, Recall: {recall}, Precision: {precision}")
#Best Hyperparameters: Number of Trees: 50, Subset Feature Size: 4, Sample Size: 500
#Full Model Metrics - Accuracy: 0.5151553852703278, Test Accuracy: 0.5151553852703278, R-squared: -0.4090676883780333, F-1 Score: 0.411973811123256, AUC: nan, Recall: 0.5151553852703278, Precision: 0.3435367047250203    

"""
"""


























"""
import pandas as pd
from DecisionTree import *
import random
from Metrics import *



def aggregate_predictions(trees, test_data):
    # Aggregate predictions from all trees
    predictions = [error(tree, test_data)[1] for tree in trees]
    predictions = np.array(predictions).T
    predictions = np.apply_along_axis(replace_none, axis=1, arr=predictions)
    if predictions.shape[1] == 1:
        # Binary classification: Majority vote
        majority_vote = np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[0][np.argmax(np.unique(x, return_counts=True)[1])], axis=1, arr=predictions)
        return majority_vote
    else:
        # Multiclass or regression: Average prediction
        return np.mean(predictions, axis=1)

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
    columns_to_drop = ['diag_1', 'diag_2', 'diag_3']
    data.drop(columns=columns_to_drop, inplace=True)
    X = data.drop(end_goal, axis=1)
    y = data[end_goal]
    return X, y

def run_tree_on_diabetes_data(file_path, end_label):
    # Load the new dataset
    data = pd.read_csv(file_path)
    data = preprocess_and_rearrange(data, end_label)
    X, y = preprocess_diabetes_data(data, end_goal=end_label)
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.1)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    #attributes = list(X_train.columns) 
    #attribute_indices = [i for i in range(len(X_train.columns))]
    feature_subset_sizes =[6] #[2, 4, 6]
    max_trees = 10 # 10 for testing and 500 for full run
    #n_iterations = 100 # 1 for testing and 100 for full run
    n_trees = max_trees 
    sample_size = 10000 
    # a for loop for covering different subset sizes, in the loop we build the max_trees sized forests (it is called by the bias_variance_decomposition_rf, which is in the main DecisionTree library)
    # and compute the bias, variance, and squared error for them and do the plotting as asked in the problem
    sampled_train = train.sample(n=sample_size, replace=False)
    train_errors_rf, test_errors_rf, trees = random_forest_data(sampled_train, test, end_label, n_trees, feature_subset_sizes[0])
    test_acc, r_squared, f1, auc, recall, precision, predict = metrics_report(trees, test)
    print("Test Accuracy: ", test_acc, "R-squared: ", r_squared, "F-1 Score: ", f1, "Area Under the Curve (AUC) for Binary Variables: ", auc, "Recall: ", recall, "Model Precision: ", precision)

print("Prediction for Readmittion with Random Forest: ")
run_tree_on_diabetes_data('smote_re.csv', 'readmitted')
#print("Prediction for Time in Hospital with Random Forest: ")
#run_tree_on_diabetes_data('smote_time.csv', 'time_in_hospital')
"""