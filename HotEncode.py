import pandas as pd

def preprocess_for_svm(file_path, label_col, numerical_cols, ordinal_cols, columns_to_drop):
    # Load dataset
    data = pd.read_csv(file_path)

    # Remove specified columns
    data.drop(columns=columns_to_drop, inplace=True)

    # Separate label column
    label = data[label_col]
    data.drop(columns=[label_col], inplace=True)

    # One-hot encode non-numerical and non-ordinal columns
    categorical_cols = data.columns.difference(numerical_cols + ordinal_cols)
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True, dtype=int)

    # Add label column back as the last column
    data[label_col] = label

    return data

# Numerical columns
numerical_cols = ['num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 
                  'number_emergency', 'number_inpatient', 'number_diagnoses', 'service_utilization', 
                  'number_of_medications', 'number_of_medication_changes']

# Ordinal columns (already properly coded)
ordinal_cols = ['age']

# Process 'smote_re.csv'
smote_re_data = preprocess_for_svm('smote_re.csv', 'readmitted', numerical_cols, ordinal_cols, ['diag_1', 'diag_2', 'diag_3'])
smote_re_data.to_csv('smote_re_sv.csv', index=False)

# Process 'smote_time.csv'
smote_time_data = preprocess_for_svm('smote_time.csv', 'time_in_hospital', numerical_cols, ordinal_cols, ['diag_1', 'diag_2', 'diag_3'])
smote_time_data.to_csv('smote_time_sv.csv', index=False)
