from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

def train_decision_tree_and_report_metrics(file_path, target_column):
    data = pd.read_csv(file_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    decision_tree_model = DecisionTreeClassifier(random_state=42)
    decision_tree_model.fit(X_train, y_train)
    y_pred = decision_tree_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    if len(y.unique()) == 2:
        y_pred_proba = decision_tree_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
    else:
        auc_score = None
    return accuracy, clf_report, conf_matrix, auc_score

readmitted_file_path = 'smote_re.csv'
time_in_hospital_file_path = 'smote_time.csv'

accuracy_readmitted, clf_report_readmitted, conf_matrix_readmitted, auc_score_readmitted = train_decision_tree_and_report_metrics(readmitted_file_path, 'readmitted')
print(f"Metrics for 'readmitted' with Decision Tree:\nAccuracy: {accuracy_readmitted}\nClassification Report:\n{clf_report_readmitted}\nConfusion Matrix:\n{conf_matrix_readmitted}\nAUC Score: {auc_score_readmitted if auc_score_readmitted is not None else 'N/A'}\n")

accuracy_time_in_hospital, clf_report_time_in_hospital, conf_matrix_time_in_hospital, auc_score_time_in_hospital = train_decision_tree_and_report_metrics(time_in_hospital_file_path, 'time_in_hospital')
print(f"Metrics for 'time_in_hospital' with Decision Tree:\nAccuracy: {accuracy_time_in_hospital}\nClassification Report:\n{clf_report_time_in_hospital}\nConfusion Matrix:\n{conf_matrix_time_in_hospital}\nAUC Score: {auc_score_time_in_hospital if auc_score_time_in_hospital is not None else 'N/A'}")

"""
Metrics for 'readmitted' with Decision Tree:
Accuracy: 0.8720916727962525
Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.86      0.87     25725
           1       0.86      0.88      0.87     25937

    accuracy                           0.87     51662
   macro avg       0.87      0.87      0.87     51662
weighted avg       0.87      0.87      0.87     51662

Confusion Matrix:
[[22114  3611]
 [ 2997 22940]]
AUC Score: 0.8720407470100329

Metrics for 'time_in_hospital' with Decision Tree:
Accuracy: 0.5826829383662079
Classification Report:
              precision    recall  f1-score   support

           0       0.68      0.67      0.68     14144
           1       0.45      0.46      0.46     14105
           2       0.62      0.61      0.62     14033

    accuracy                           0.58     42282
   macro avg       0.58      0.58      0.58     42282
weighted avg       0.58      0.58      0.58     42282

Confusion Matrix:
[[9546 3487 1111]
 [3438 6512 4155]
 [1081 4373 8579]]
AUC Score: N/A
"""