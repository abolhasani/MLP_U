import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

def train_logistic_regression_and_report_metrics(file_path, target_column):
    data = pd.read_csv(file_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
    log_reg_model.fit(X_train, y_train)
    y_pred = log_reg_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    if len(y.unique()) == 2:
        y_pred_proba = log_reg_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
    else:
        auc_score = None
    return accuracy, clf_report, conf_matrix, auc_score

readmitted_file_path = 'smote_re.csv'
time_in_hospital_file_path = 'smote_time.csv'

accuracy_readmitted, clf_report_readmitted, conf_matrix_readmitted, auc_score_readmitted = train_logistic_regression_and_report_metrics(readmitted_file_path, 'readmitted')
print(f"Metrics for 'readmitted':\nAccuracy: {accuracy_readmitted}\nClassification Report:\n{clf_report_readmitted}\nConfusion Matrix:\n{conf_matrix_readmitted}\nAUC Score: {auc_score_readmitted}\n")

accuracy_time_in_hospital, clf_report_time_in_hospital, conf_matrix_time_in_hospital, auc_score_time_in_hospital = train_logistic_regression_and_report_metrics(time_in_hospital_file_path, 'time_in_hospital')
print(f"Metrics for 'time_in_hospital':\nAccuracy: {accuracy_time_in_hospital}\nClassification Report:\n{clf_report_time_in_hospital}\nConfusion Matrix:\n{conf_matrix_time_in_hospital}\nAUC Score: {auc_score_time_in_hospital}")

"""
Metrics for 'readmitted':
Accuracy: 0.7769346908753049
Classification Report:
              precision    recall  f1-score   support

           0       0.77      0.79      0.78     25725
           1       0.79      0.76      0.77     25937

    accuracy                           0.78     51662
   macro avg       0.78      0.78      0.78     51662
weighted avg       0.78      0.78      0.78     51662

Confusion Matrix:
[[20325  5400]
 [ 6124 19813]]
AUC Score: 0.8493790496992919

Metrics for 'time_in_hospital':
Accuracy: 0.5809091339104111
Classification Report:
              precision    recall  f1-score   support

           0       0.64      0.68      0.66     14144
           1       0.46      0.36      0.40     14105
           2       0.61      0.70      0.65     14033

    accuracy                           0.58     42282
   macro avg       0.57      0.58      0.57     42282
weighted avg       0.57      0.58      0.57     42282

Confusion Matrix:
[[9596 2879 1669]
 [4265 5084 4756]
 [1028 3123 9882]]
AUC Score: None
"""