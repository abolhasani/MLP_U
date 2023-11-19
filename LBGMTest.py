import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import pandas as pd

def train_lightgbm_and_report_metrics(file_path, target_column):
    data = pd.read_csv(file_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    lgbm_model = lgb.LGBMClassifier(random_state=42)
    lgbm_model.fit(X_train, y_train)
    y_pred = lgbm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    if len(y.unique()) == 2:
        y_pred_proba = lgbm_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
    else:
        auc_score = None
    return accuracy, clf_report, conf_matrix, auc_score

readmitted_file_path = 'smote_re.csv'
time_in_hospital_file_path = 'smote_time.csv'

accuracy_readmitted, clf_report_readmitted, conf_matrix_readmitted, auc_score_readmitted = train_lightgbm_and_report_metrics(readmitted_file_path, 'readmitted')
print(f"Metrics for 'readmitted':\nAccuracy: {accuracy_readmitted}\nClassification Report:\n{clf_report_readmitted}\nConfusion Matrix:\n{conf_matrix_readmitted}\nAUC Score: {auc_score_readmitted}\n")

accuracy_time_in_hospital, clf_report_time_in_hospital, conf_matrix_time_in_hospital, auc_score_time_in_hospital = train_lightgbm_and_report_metrics(time_in_hospital_file_path, 'time_in_hospital')
print(f"Metrics for 'time_in_hospital':\nAccuracy: {accuracy_time_in_hospital}\nClassification Report:\n{clf_report_time_in_hospital}\nConfusion Matrix:\n{conf_matrix_time_in_hospital}\nAUC Score: {auc_score_time_in_hospital}")

"""
[LightGBM] [Info] Number of positive: 60165, number of negative: 60377
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.008472 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 3431
[LightGBM] [Info] Number of data points in the train set: 120542, number of used features: 36
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.499121 -> initscore=-0.003517
[LightGBM] [Info] Start training from score -0.003517
Metrics for 'readmitted':
Accuracy: 0.9329100692965816
Classification Report:
              precision    recall  f1-score   support

           0       0.88      1.00      0.94     25725
           1       1.00      0.87      0.93     25937

    accuracy                           0.93     51662
   macro avg       0.94      0.93      0.93     51662
weighted avg       0.94      0.93      0.93     51662

Confusion Matrix:
[[25635    90]
 [ 3376 22561]]
AUC Score: 0.9580310202942594

[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.004724 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 3423
[LightGBM] [Info] Number of data points in the train set: 98658, number of used features: 36
[LightGBM] [Info] Start training from score -1.100134
[LightGBM] [Info] Start training from score -1.098947
[LightGBM] [Info] Start training from score -1.096759
Metrics for 'time_in_hospital':
Accuracy: 0.6875975592450688
Classification Report:
              precision    recall  f1-score   support

           0       0.78      0.77      0.77     14144
           1       0.56      0.51      0.54     14105
           2       0.71      0.78      0.74     14033

    accuracy                           0.69     42282
   macro avg       0.68      0.69      0.68     42282
weighted avg       0.68      0.69      0.68     42282

Confusion Matrix:
[[10859  2780   505]
 [ 2880  7235  3990]
 [  161  2893 10979]]
AUC Score: None
"""
