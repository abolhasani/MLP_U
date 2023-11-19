from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

def train_knn_and_report_metrics(file_path, target_column, n_neighbors=4):
    data = pd.read_csv(file_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = X_train.values
    X_test = X_test.values
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    if len(y.unique()) == 2:
        y_pred_proba = knn_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
    else:
        auc_score = None

    return accuracy, clf_report, conf_matrix, auc_score

readmitted_file_path = 'smote_re.csv'
time_in_hospital_file_path = 'smote_time.csv'

accuracy_readmitted, clf_report_readmitted, conf_matrix_readmitted, auc_score_readmitted = train_knn_and_report_metrics(readmitted_file_path, 'readmitted', n_neighbors=4)
print(f"Metrics for 'readmitted' with KNN (4 Neighbors):\nAccuracy: {accuracy_readmitted}\nClassification Report:\n{clf_report_readmitted}\nConfusion Matrix:\n{conf_matrix_readmitted}\nAUC Score: {auc_score_readmitted if auc_score_readmitted is not None else 'N/A'}\n")

accuracy_time_in_hospital, clf_report_time_in_hospital, conf_matrix_time_in_hospital, auc_score_time_in_hospital = train_knn_and_report_metrics(time_in_hospital_file_path, 'time_in_hospital', n_neighbors=4)
print(f"Metrics for 'time_in_hospital' with KNN (4 Neighbors):\nAccuracy: {accuracy_time_in_hospital}\nClassification Report:\n{clf_report_time_in_hospital}\nConfusion Matrix:\n{conf_matrix_time_in_hospital}\nAUC Score: {auc_score_time_in_hospital if auc_score_time_in_hospital is not None else 'N/A'}")

"""
Metrics for 'readmitted' with KNN (4 Neighbors):
Accuracy: 0.8297975300994929
Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.74      0.81     25725
           1       0.78      0.91      0.84     25937

    accuracy                           0.83     51662
   macro avg       0.84      0.83      0.83     51662
weighted avg       0.84      0.83      0.83     51662

Confusion Matrix:
[[19137  6588]
 [ 2205 23732]]
AUC Score: 0.9005202559105148

Metrics for 'time_in_hospital' with KNN (4 Neighbors):
Accuracy: 0.6064992195260395
Classification Report:
              precision    recall  f1-score   support

           0       0.63      0.87      0.73     14144
           1       0.54      0.53      0.53     14105
           2       0.65      0.42      0.51     14033

    accuracy                           0.61     42282
   macro avg       0.61      0.61      0.59     42282
weighted avg       0.61      0.61      0.59     42282

Confusion Matrix:
[[12321  1148   675]
 [ 4145  7446  2514]
 [ 3010  5146  5877]]
AUC Score: N/A
"""