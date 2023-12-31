from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import pandas as pd

def train_svm_and_report_metrics(file_path, target_column):
    data = pd.read_csv(file_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = X_train.values
    X_test = X_test.values
    svm_model = SVC(probability=True, max_iter =10,  random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    if len(y.unique()) == 2:
        y_pred_proba = svm_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
    else:
        auc_score = None
    return accuracy, clf_report, conf_matrix, auc_score

readmitted_file_path = 'smote_re_sv.csv'
time_in_hospital_file_path = 'smote_time_sv.csv'

accuracy_readmitted, clf_report_readmitted, conf_matrix_readmitted, auc_score_readmitted = train_svm_and_report_metrics(readmitted_file_path, 'readmitted')
print(f"Metrics for 'readmitted':\nAccuracy: {accuracy_readmitted}\nClassification Report:\n{clf_report_readmitted}\nConfusion Matrix:\n{conf_matrix_readmitted}\nAUC Score: {auc_score_readmitted if auc_score_readmitted is not None else 'N/A'}\n")

accuracy_time_in_hospital, clf_report_time_in_hospital, conf_matrix_time_in_hospital, auc_score_time_in_hospital = train_svm_and_report_metrics(time_in_hospital_file_path, 'time_in_hospital')
print(f"Metrics for 'time_in_hospital':\nAccuracy: {accuracy_time_in_hospital}\nClassification Report:\n{clf_report_time_in_hospital}\nConfusion Matrix:\n{conf_matrix_time_in_hospital}\nAUC Score: {auc_score_time_in_hospital if auc_score_time_in_hospital is not None else 'N/A'}")

"""
Metrics for 'readmitted':
Accuracy: 0.5284541829584608
Classification Report:
              precision    recall  f1-score   support

           0       0.54      0.34      0.42     25725
           1       0.52      0.72      0.60     25937

    accuracy                           0.53     51662
   macro avg       0.53      0.53      0.51     51662
weighted avg       0.53      0.53      0.51     51662

Confusion Matrix:
[[ 8754 16971]
 [ 7390 18547]]
AUC Score: 0.5390354957795058

Metrics for 'time_in_hospital':
Accuracy: 0.37498226195544204
Classification Report:
              precision    recall  f1-score   support

           0       0.43      0.30      0.36     14144
           1       0.34      0.74      0.47     14105
           2       0.69      0.08      0.14     14033

    accuracy                           0.37     42282
   macro avg       0.49      0.37      0.32     42282
weighted avg       0.49      0.37      0.32     42282

Confusion Matrix:
[[ 4283  9716   145]
 [ 3258 10501   346]
 [ 2371 10591  1071]]
AUC Score: N/A
"""