from sklearn.linear_model import Perceptron
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

def train_perceptron_and_report_metrics(file_path, target_column):
    data = pd.read_csv(file_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    perceptron_model = Perceptron(random_state=42)
    perceptron_model.fit(X_train, y_train)
    y_pred = perceptron_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    if len(y.unique()) == 2 and hasattr(perceptron_model, "predict_proba"):
        y_pred_proba = perceptron_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
    else:
        auc_score = None
    return accuracy, clf_report, conf_matrix, auc_score

readmitted_file_path = 'smote_re.csv'

accuracy_readmitted, clf_report_readmitted, conf_matrix_readmitted, auc_score_readmitted = train_perceptron_and_report_metrics(readmitted_file_path, 'readmitted')
print(f"Metrics for 'readmitted' with Perceptron:\nAccuracy: {accuracy_readmitted}\nClassification Report:\n{clf_report_readmitted}\nConfusion Matrix:\n{conf_matrix_readmitted}\nAUC Score: {auc_score_readmitted if auc_score_readmitted is not None else 'N/A'}")

"""
Metrics for 'readmitted' with Perceptron:
Accuracy: 0.6442839998451473
Classification Report:
              precision    recall  f1-score   support

           0       0.59      0.92      0.72     25725
           1       0.82      0.37      0.51     25937

    accuracy                           0.64     51662
   macro avg       0.71      0.65      0.62     51662
weighted avg       0.71      0.64      0.62     51662

Confusion Matrix:
[[23669  2056]
 [16321  9616]]
AUC Score: N/A
"""