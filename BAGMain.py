import pandas as pd
from DecisionTree import ID3, custom_train_test_split, error, bagged_trees_data
from Metrics import *
import pandas as pd

# move the label column to the end of the DataFrame which wasn't handled in the preprocessing file at first
def preprocess_and_rearrange(data, label_column):
    label = data[label_column]
    data = data.drop(label_column, axis=1)
    data[label_column] = label
    return data

def preprocess_diabetes_data(data, end_goal):
    for column in ['num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'service_utilization', 'number_of_medications', 'number_of_medication_changes']:
        median_val = data[column].median()
        data[column] = (data[column] > median_val).astype(int)
    X = data.drop(end_goal, axis=1)
    y = data[end_goal]
    return X, y

def run_tree_on_diabetes_data(file_path, end_label):
    data = pd.read_csv(file_path)
    data = preprocess_and_rearrange(data, end_label)
    X, y = preprocess_diabetes_data(data, end_goal=end_label)
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.1)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    #attributes = list(X_train.columns) 
    #attribute_indices = [i for i in range(len(X_train.columns))]
    max_trees = 10 
    training_error, _, tree = bagged_trees_data(train, test, max_trees, end_label, 'ME')
    #print("Training set Error: ", training_error)
    test_acc, r_squared, f1, auc, recall, precision, predict = metrics_report(tree, test)
    print("Test Accuracy: ", test_acc, "R-squared: ", r_squared, "F-1 Score: ", f1, "Area Under the Curve (AUC) for Binary Variables: ", auc, "Recall: ", recall, "Model Precision: ", precision)

print("Prediction for Readmittion with Bagged Trees:")
run_tree_on_diabetes_data('smote_re.csv', 'readmitted')
print("Prediction for Time in Hospital with Bagged Trees: ")
run_tree_on_diabetes_data('smote_time.csv', 'time_in_hospital')

"""
Prediction for Readmittion with Bagged Trees:
Test Accuracy:  0.8814169570267131 R-squared:  0.00902308714345934 F-1 Score:  0.7481084055842648 Area Under the Curve (AUC) for Binary Variables:  0.7515164159707979 Recall:  0.7515164159707979 Model Precision:  0.7683993003821986
Prediction for Time in Hospital with Bagged Trees:
Test Accuracy:  0.7498226195544203 R-squared:  -0.5556756382293184 F-1 Score:  0.46947709135164173 Area Under the Curve (AUC) for Binary Variables:  None Recall:  0.48256996353044895 Model Precision:  0.5241387853021152
"""