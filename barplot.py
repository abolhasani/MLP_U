import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# function to compute feature importance and classification metrics
def compute_feature_importance_and_metrics(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    feature_names = X.columns
    feature_importances = pd.Series(importances, index=feature_names)
    sorted_feature_importances = feature_importances.sort_values(ascending=False)[:15]
    plt.figure(figsize=(10, 6))
    sorted_feature_importances.plot(kind='bar')
    plt.title(f'Top 15 Feature Importances for {target_column}')
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.show()
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, clf_report, conf_matrix

# Load the datasets
readmitted_data = pd.read_csv('smote_re.csv')
time_in_hospital_data = pd.read_csv('smote_time.csv')

accuracy_readmitted, clf_report_readmitted, conf_matrix_readmitted = compute_feature_importance_and_metrics(readmitted_data, 'readmitted')
#print(f"Metrics for 'readmitted':\nAccuracy: {accuracy_readmitted}\nClassification Report:\n{clf_report_readmitted}\nConfusion Matrix:\n{conf_matrix_readmitted}\n")

accuracy_time_in_hospital, clf_report_time_in_hospital, conf_matrix_time_in_hospital = compute_feature_importance_and_metrics(time_in_hospital_data, 'time_in_hospital')
#print(f"Metrics for 'time_in_hospital':\nAccuracy: {accuracy_time_in_hospital}\nClassification Report:\n{clf_report_time_in_hospital}\nConfusion Matrix:\n{conf_matrix_time_in_hospital}")

