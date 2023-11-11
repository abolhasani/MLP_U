from DecisionTree import ID3, custom_train_test_split, error
from Metrics import *
import pandas as pd

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
    X = data.drop(end_goal, axis=1)
    y = data[end_goal]

    return X, y

def run_tree_on_diabetes_data(file_path, end_label):
    # Load the new dataset
    data = pd.read_csv(file_path)
    data = preprocess_and_rearrange(data, end_label)
    X, y = preprocess_diabetes_data(data, end_goal=end_label)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.1)

    # Preparing for the decision tree algorithm
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    attributes = list(X_train.columns) 
    attribute_indices = [i for i in range(len(X_train.columns))]
    # Initializing depth, heuristic methods, and result data frame
    #result = {'train error': [], 'test error': []}
            
    tree = ID3(train.values, attribute_indices, end_label, heuristic='HS')

    #training_error, _ = error(tree, train)
    #print("Training set Error: ", training_error)
    #testing_error, testing_predict = error(tree, test)
    #result['train error'].append(training_error)
    #result['test error'].append(testing_error)

    #result_df = pd.DataFrame(result)
    #print(result_df)
    test_acc, r_squared, f1, auc, recall, precision, predict = metrics_report(tree, test)
    print("Test Accuracy: ", test_acc, "R-squared: ", r_squared, "F-1 Score: ", f1, "Area Under the Curve (AUC) for Binary Variables: ", auc, "Recall: ", recall, "Model Precision: ", precision)

# Replace 'path_to_your_new_data.csv' with the actual path to your new dataset
print("Prediction for Readmittion with Decision Trees:")
run_tree_on_diabetes_data('smote_re.csv', 'readmitted')
print("Prediction for Time in Hospital with Decision Trees: ")
run_tree_on_diabetes_data('smote_time.csv', 'time_in_hospital')
