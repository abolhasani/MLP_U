from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd

file_path = 'preproc.csv'  
data = pd.read_csv(file_path)

# Apply Label Encoding to categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for column in categorical_columns:
    data[column] = le.fit_transform(data[column])
print("Original 'readmitted' distribution:\n", data['readmitted'].value_counts())

# SMOTE for 'readmitted'
X_readmitted = data.drop(columns=['readmitted'])
y_readmitted = data['readmitted']
smote_readmitted = SMOTE()
X_readmitted_resampled, y_readmitted_resampled = smote_readmitted.fit_resample(X_readmitted, y_readmitted)
print("Resampled 'readmitted' distribution:\n", y_readmitted_resampled.value_counts())
readmitted_resampled_data = pd.concat([X_readmitted_resampled, y_readmitted_resampled], axis=1)
readmitted_resampled_data.to_csv('smote_re.csv', index=False)

# 'time_in_hospital'
print("Original 'time_in_hospital' distribution:\n", data['time_in_hospital'].value_counts())
X_time_in_hospital = data.drop(columns=['time_in_hospital'])
y_time_in_hospital = data['time_in_hospital']
smote_time_in_hospital = SMOTE()
X_time_in_hospital_resampled, y_time_in_hospital_resampled = smote_time_in_hospital.fit_resample(X_time_in_hospital, y_time_in_hospital)
print("Resampled 'time_in_hospital' distribution:\n", y_time_in_hospital_resampled.value_counts())
time_in_hospital_resampled_data = pd.concat([X_time_in_hospital_resampled, y_time_in_hospital_resampled], axis=1)
time_in_hospital_resampled_data.to_csv('smote_time.csv', index=False)

"""
Original 'readmitted' distribution:
 readmitted
0    86102
1    10934
Name: count, dtype: int64
Resampled 'readmitted' distribution:
 readmitted
0    86102
1    86102
Name: count, dtype: int64
Original 'time_in_hospital' distribution:
 time_in_hospital
2    46980
1    36147
0    13909
Name: count, dtype: int64
Resampled 'time_in_hospital' distribution:
 time_in_hospital
2    46980
1    46980
0    46980
Name: count, dtype: int64
"""