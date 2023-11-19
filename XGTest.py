import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

def train_xgboost_and_report_metrics(file_path, target_column):
    data = pd.read_csv(file_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    if len(y.unique()) == 2:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
    else:
        auc_score = None
    return accuracy, clf_report, conf_matrix, auc_score

readmitted_file_path = 'smote_re.csv'
time_in_hospital_file_path = 'smote_time.csv'

accuracy_readmitted, clf_report_readmitted, conf_matrix_readmitted, auc_score_readmitted = train_xgboost_and_report_metrics(readmitted_file_path, 'readmitted')
print(f"Metrics for 'readmitted':\nAccuracy: {accuracy_readmitted}\nClassification Report:\n{clf_report_readmitted}\nConfusion Matrix:\n{conf_matrix_readmitted}\nAUC Score: {auc_score_readmitted}\n")

accuracy_time_in_hospital, clf_report_time_in_hospital, conf_matrix_time_in_hospital, auc_score_time_in_hospital = train_xgboost_and_report_metrics(time_in_hospital_file_path, 'time_in_hospital')
print(f"Metrics for 'time_in_hospital':\nAccuracy: {accuracy_time_in_hospital}\nClassification Report:\n{clf_report_time_in_hospital}\nConfusion Matrix:\n{conf_matrix_time_in_hospital}\nAUC Score: {auc_score_time_in_hospital}")

"""
Metrics for 'readmitted':
Accuracy: 0.9326584336649761
Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.99      0.94     25725
           1       0.99      0.87      0.93     25937

    accuracy                           0.93     51662
   macro avg       0.94      0.93      0.93     51662
weighted avg       0.94      0.93      0.93     51662

Confusion Matrix:
[[25577   148]
 [ 3331 22606]]
AUC Score: 0.9567943427846191

Metrics for 'time_in_hospital':
Accuracy: 0.6961354713589707
Classification Report:
              precision    recall  f1-score   support

           0       0.79      0.77      0.78     14144
           1       0.57      0.53      0.55     14105
           2       0.72      0.78      0.75     14033

    accuracy                           0.70     42282
   macro avg       0.69      0.70      0.69     42282
weighted avg       0.69      0.70      0.69     42282

Confusion Matrix:
[[10958  2709   477]
 [ 2723  7528  3854]
 [  169  2916 10948]]
AUC Score: None
"""