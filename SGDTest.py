from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

# Function to train SGD Classifier and report metrics
def train_sgd_classifier_and_report_metrics(file_path, target_column):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize SGD Classifier
    sgd_model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    sgd_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = sgd_model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # AUC for binary classification
    if len(y.unique()) == 2 and hasattr(sgd_model, "predict_proba"):
        y_pred_proba = sgd_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
    else:
        auc_score = None

    return accuracy, clf_report, conf_matrix, auc_score

# Paths to the CSV files
readmitted_file_path = 'smote_re_sv.csv'
time_in_hospital_file_path = 'smote_time_sv.csv'

# Train SGD Classifier and report metrics for 'readmitted'
accuracy_readmitted, clf_report_readmitted, conf_matrix_readmitted, auc_score_readmitted = train_sgd_classifier_and_report_metrics(readmitted_file_path, 'readmitted')
print(f"Metrics for 'readmitted' with SGD Classifier:\nAccuracy: {accuracy_readmitted}\nClassification Report:\n{clf_report_readmitted}\nConfusion Matrix:\n{conf_matrix_readmitted}\nAUC Score: {auc_score_readmitted if auc_score_readmitted is not None else 'N/A'}\n")

# Train SGD Classifier and report metrics for 'time_in_hospital'
accuracy_time_in_hospital, clf_report_time_in_hospital, conf_matrix_time_in_hospital, auc_score_time_in_hospital = train_sgd_classifier_and_report_metrics(time_in_hospital_file_path, 'time_in_hospital')
print(f"Metrics for 'time_in_hospital' with SGD Classifier:\nAccuracy: {accuracy_time_in_hospital}\nClassification Report:\n{clf_report_time_in_hospital}\nConfusion Matrix:\n{conf_matrix_time_in_hospital}\nAUC Score: {auc_score_time_in_hospital if auc_score_time_in_hospital is not None else 'N/A'}")

