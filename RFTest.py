import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

def train_random_forest_and_report_metrics(file_path, target_column):
    data = pd.read_csv(file_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    if len(y.unique()) == 2:
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
    else:
        auc_score = None
    return accuracy, clf_report, conf_matrix, auc_score

readmitted_file_path = 'smote_re.csv'
time_in_hospital_file_path = 'smote_time.csv'

accuracy_readmitted, clf_report_readmitted, conf_matrix_readmitted, auc_score_readmitted = train_random_forest_and_report_metrics(readmitted_file_path, 'readmitted')
print(f"Metrics for 'readmitted':\nAccuracy: {accuracy_readmitted}\nClassification Report:\n{clf_report_readmitted}\nConfusion Matrix:\n{conf_matrix_readmitted}\nAUC Score: {auc_score_readmitted}\n")

accuracy_time_in_hospital, clf_report_time_in_hospital, conf_matrix_time_in_hospital, auc_score_time_in_hospital = train_random_forest_and_report_metrics(time_in_hospital_file_path, 'time_in_hospital')
print(f"Metrics for 'time_in_hospital':\nAccuracy: {accuracy_time_in_hospital}\nClassification Report:\n{clf_report_time_in_hospital}\nConfusion Matrix:\n{conf_matrix_time_in_hospital}\nAUC Score: {auc_score_time_in_hospital}")

"""
Metrics for 'readmitted':
Accuracy: 0.9321358058147188
Classification Report:
              precision    recall  f1-score   support

           0       0.88      1.00      0.94     25725
           1       0.99      0.87      0.93     25937

    accuracy                           0.93     51662
   macro avg       0.94      0.93      0.93     51662
weighted avg       0.94      0.93      0.93     51662

Confusion Matrix:
[[25608   117]
 [ 3389 22548]]
AUC Score: 0.9598303948945888

Metrics for 'time_in_hospital':
Accuracy: 0.6927770682559955
Classification Report:
              precision    recall  f1-score   support

           0       0.79      0.80      0.80     14144
           1       0.58      0.51      0.54     14105
           2       0.69      0.77      0.73     14033

    accuracy                           0.69     42282
   macro avg       0.69      0.69      0.69     42282
weighted avg       0.69      0.69      0.69     42282

Confusion Matrix:
[[11305  2189   650]
 [ 2750  7248  4107]
 [  192  3102 10739]]
AUC Score: None
"""