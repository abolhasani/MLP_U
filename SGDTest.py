from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

def train_sgd_classifier_and_report_metrics(file_path, target_column):
    data = pd.read_csv(file_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    sgd_model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    sgd_model.fit(X_train, y_train)
    y_pred = sgd_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    if len(y.unique()) == 2 and hasattr(sgd_model, "predict_proba"):
        y_pred_proba = sgd_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
    else:
        auc_score = None
    return accuracy, clf_report, conf_matrix, auc_score

readmitted_file_path = 'smote_re_sv.csv'
time_in_hospital_file_path = 'smote_time_sv.csv'

accuracy_readmitted, clf_report_readmitted, conf_matrix_readmitted, auc_score_readmitted = train_sgd_classifier_and_report_metrics(readmitted_file_path, 'readmitted')
print(f"Metrics for 'readmitted' with SGD Classifier:\nAccuracy: {accuracy_readmitted}\nClassification Report:\n{clf_report_readmitted}\nConfusion Matrix:\n{conf_matrix_readmitted}\nAUC Score: {auc_score_readmitted if auc_score_readmitted is not None else 'N/A'}\n")

accuracy_time_in_hospital, clf_report_time_in_hospital, conf_matrix_time_in_hospital, auc_score_time_in_hospital = train_sgd_classifier_and_report_metrics(time_in_hospital_file_path, 'time_in_hospital')
print(f"Metrics for 'time_in_hospital' with SGD Classifier:\nAccuracy: {accuracy_time_in_hospital}\nClassification Report:\n{clf_report_time_in_hospital}\nConfusion Matrix:\n{conf_matrix_time_in_hospital}\nAUC Score: {auc_score_time_in_hospital if auc_score_time_in_hospital is not None else 'N/A'}")

"""
Metrics for 'readmitted' with SGD Classifier:
Accuracy: 0.8500638767372537
Classification Report:
              precision    recall  f1-score   support

           0       0.78      0.97      0.87     25725
           1       0.96      0.73      0.83     25937

    accuracy                           0.85     51662
   macro avg       0.87      0.85      0.85     51662
weighted avg       0.87      0.85      0.85     51662

Confusion Matrix:
[[25014   711]
 [ 7035 18902]]
AUC Score: N/A

Metrics for 'time_in_hospital' with SGD Classifier:
Accuracy: 0.5841965848351545
Classification Report:
              precision    recall  f1-score   support

           0       0.66      0.79      0.72     14144
           1       0.55      0.03      0.05     14105
           2       0.53      0.94      0.68     14033

    accuracy                           0.58     42282
   macro avg       0.58      0.58      0.48     42282
weighted avg       0.58      0.58      0.48     42282

Confusion Matrix:
[[11140   185  2819]
 [ 4887   353  8865]
 [  726    99 13208]]
AUC Score: N/A
"""