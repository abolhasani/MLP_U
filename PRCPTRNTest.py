from sklearn.linear_model import Perceptron
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

# Function to train Perceptron and report metrics
def train_perceptron_and_report_metrics(file_path, target_column):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize Perceptron classifier
    perceptron_model = Perceptron(random_state=42)
    perceptron_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = perceptron_model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # AUC for binary classification
    if len(y.unique()) == 2 and hasattr(perceptron_model, "predict_proba"):
        y_pred_proba = perceptron_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
    else:
        auc_score = None

    return accuracy, clf_report, conf_matrix, auc_score

# Path to the CSV file
readmitted_file_path = 'smote_re.csv'

# Train Perceptron and report metrics for 'readmitted'
accuracy_readmitted, clf_report_readmitted, conf_matrix_readmitted, auc_score_readmitted = train_perceptron_and_report_metrics(readmitted_file_path, 'readmitted')
print(f"Metrics for 'readmitted' with Perceptron:\nAccuracy: {accuracy_readmitted}\nClassification Report:\n{clf_report_readmitted}\nConfusion Matrix:\n{conf_matrix_readmitted}\nAUC Score: {auc_score_readmitted if auc_score_readmitted is not None else 'N/A'}")

