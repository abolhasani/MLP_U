from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd

# Load the dataset
file_path = 'preproc.csv'  # Replace with the correct path to your dataset
data = pd.read_csv(file_path)

# Apply Label Encoding to categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for column in categorical_columns:
    data[column] = le.fit_transform(data[column])

# Print original distribution for 'readmitted'
print("Original 'readmitted' distribution:\n", data['readmitted'].value_counts())

# SMOTE for 'readmitted'
X_readmitted = data.drop(columns=['readmitted'])
y_readmitted = data['readmitted']
smote_readmitted = SMOTE()
X_readmitted_resampled, y_readmitted_resampled = smote_readmitted.fit_resample(X_readmitted, y_readmitted)

# Print resampled distribution for 'readmitted'
print("Resampled 'readmitted' distribution:\n", y_readmitted_resampled.value_counts())

readmitted_resampled_data = pd.concat([X_readmitted_resampled, y_readmitted_resampled], axis=1)
readmitted_resampled_data.to_csv('smote_re.csv', index=False)

# Print original distribution for 'time_in_hospital'
print("Original 'time_in_hospital' distribution:\n", data['time_in_hospital'].value_counts())

# SMOTE for 'time_in_hospital'
X_time_in_hospital = data.drop(columns=['time_in_hospital'])
y_time_in_hospital = data['time_in_hospital']
smote_time_in_hospital = SMOTE()
X_time_in_hospital_resampled, y_time_in_hospital_resampled = smote_time_in_hospital.fit_resample(X_time_in_hospital, y_time_in_hospital)

# Print resampled distribution for 'time_in_hospital'
print("Resampled 'time_in_hospital' distribution:\n", y_time_in_hospital_resampled.value_counts())

time_in_hospital_resampled_data = pd.concat([X_time_in_hospital_resampled, y_time_in_hospital_resampled], axis=1)
time_in_hospital_resampled_data.to_csv('smote_time.csv', index=False)



"""
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

# Load the dataset
file_path = 'preproc.csv'  # Update this path to the actual file location
data = pd.read_csv(file_path)
categorical_columns = data.select_dtypes(include=['object']).columns
# Apply Label Encoding
le = LabelEncoder()
for column in categorical_columns:
    data[column] = le.fit_transform(data[column])

# For 'readmitted'
# Calculate the target number for the minority class
current_readmitted_counts = data['readmitted'].value_counts()
desired_samples_readmitted = 10000 // len(current_readmitted_counts)
sampling_strategy_readmitted = {1: desired_samples_readmitted}
X_readmitted = data.drop(columns=['readmitted'])
y_readmitted = data['readmitted']

# Apply SMOTE for 'readmitted'
smote_readmitted = SMOTE(sampling_strategy=sampling_strategy_readmitted)
X_readmitted_resampled, y_readmitted_resampled = smote_readmitted.fit_resample(X_readmitted, y_readmitted)
print("Resampled 'readmitted' distribution:\n", y_readmitted_resampled.value_counts())

# For 'time_in_hospital'
# Determine the desired sampling strategy
current_time_in_hospital_counts = data['time_in_hospital'].value_counts()
desired_samples_time_in_hospital = 10000 // len(current_time_in_hospital_counts)
sampling_strategy_time_in_hospital = {0: desired_samples_time_in_hospital, 1: desired_samples_time_in_hospital}

# Apply SMOTE for 'time_in_hospital'
smote_time_in_hospital = SMOTE(sampling_strategy=sampling_strategy_time_in_hospital)
X_time_in_hospital = data.drop(columns=['time_in_hospital'])
y_time_in_hospital = data['time_in_hospital']
X_time_in_hospital_resampled, y_time_in_hospital_resampled = smote_time_in_hospital.fit_resample(X_time_in_hospital, y_time_in_hospital)
print("Resampled 'time_in_hospital' distribution:\n", y_time_in_hospital_resampled.value_counts())


"""
"""
# Calculate the number of samples needed for each class to reach 5000 for 'readmitted'
desired_samples_readmitted = 10000
current_readmitted_counts = data['readmitted'].value_counts()
sampling_strategy_readmitted = {
    0: max(desired_samples_readmitted - current_readmitted_counts[0], 0),
    1: max(desired_samples_readmitted - current_readmitted_counts[1], 0)
}

print("Original 'readmitted' distribution:\n", data['readmitted'].value_counts())

# Apply SMOTE for 'readmitted'
X_readmitted = data.drop(columns=['readmitted'])
y_readmitted = data['readmitted']
# For 'readmitted'
desired_total_samples = 10000
current_readmitted_counts = data['readmitted'].value_counts()

# Ensure that the target number for each class is at least the current count
sampling_strategy_readmitted = {
    class_label: max(desired_total_samples // len(current_readmitted_counts), count) for class_label, count in current_readmitted_counts.items()
}

# Apply SMOTE for 'readmitted'
smote_readmitted = SMOTE(sampling_strategy=sampling_strategy_readmitted)
X_readmitted_resampled, y_readmitted_resampled = smote_readmitted.fit_resample(X_readmitted, y_readmitted)


readmitted_resampled_data = pd.concat([X_readmitted_resampled, y_readmitted_resampled], axis=1)
readmitted_resampled_data.to_csv('smote_re.csv', index=False)
print("Resampled 'readmitted' distribution:\n", y_readmitted_resampled.value_counts())



print("Original 'time_in_hospital' distribution:\n", data['time_in_hospital'].value_counts())


# Calculate the number of samples needed for each class to reach 5000 for 'time_in_hospital'
X_time_in_hospital = data.drop(columns=['time_in_hospital'])
y_time_in_hospital = data['time_in_hospital']
# Similar approach for 'time_in_hospital'
current_time_in_hospital_counts = data['time_in_hospital'].value_counts()
sampling_strategy_time_in_hospital = {
    class_label: max(desired_total_samples // len(current_time_in_hospital_counts), count) for class_label, count in current_time_in_hospital_counts.items()
}

# Apply SMOTE for 'time_in_hospital'
smote_time_in_hospital = SMOTE(sampling_strategy=sampling_strategy_time_in_hospital)
X_time_in_hospital_resampled, y_time_in_hospital_resampled = smote_time_in_hospital.fit_resample(X_time_in_hospital, y_time_in_hospital)
time_in_hospital_resampled_data = pd.concat([X_time_in_hospital_resampled, y_time_in_hospital_resampled], axis=1)
time_in_hospital_resampled_data.to_csv('smote_time.csv', index=False)

print("Resampled 'time_in_hospital' distribution:\n", y_time_in_hospital_resampled.value_counts())
"""

