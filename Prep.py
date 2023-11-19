import pandas as pd
import numpy as np

file_path = 'diabetic_data.csv'  
data = pd.read_csv(file_path)

# 'service_utilization' as the sum of 'number_outpatient', 'number_emergency', and 'number_inpatient'
data['service_utilization'] = data['number_outpatient'] + data['number_emergency'] + data['number_inpatient']

# 'number_of_medications' is the count of different medications taken by the patient
medication_columns = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 
    'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
    'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 
    'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 
    'metformin-pioglitazone'
]
data['number_of_medications'] = data[medication_columns].apply(lambda x: (x != 'No').sum(), axis=1)

# 'number_of_medication_changes' counts how many times the medications dosage was increased or decreased
change_columns = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 
    'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
    'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 
    'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 
    'metformin-pioglitazone'
]
data['number_of_medication_changes'] = data[change_columns].apply(lambda x: (x == 'Up').sum() + (x == 'Down').sum(), axis=1)


# Discretize 'time_in_hospital'
bins = [0, 4, 8, 14]
labels = ['short', 'medium', 'long']
data['time_in_hospital'] = pd.cut(data['time_in_hospital'], bins=bins, labels=labels, right=False)

# Map 'age'
age_mapping = {
    '[0-10)': 0,
    '[10-20)': 1,
    '[20-30)': 2,
    '[30-40)': 3,
    '[40-50)': 4,
    '[50-60)': 5,
    '[60-70)': 6,
    '[70-80)': 7,
    '[80-90)': 8,
    '[90-100)': 9
}
data['age'] = data['age'].map(age_mapping)

# Remove columns with high number of missing values
data.drop(columns=['weight', 'payer_code', 'medical_specialty'], inplace=True)

# Calculate skewness
def calculate_skewness(df, column):
    Y = df[column]
    Y_bar = Y.mean()
    s = Y.std()
    g1 = ((Y - Y_bar)**3).sum() / len(Y) / s**3
    return g1

numerical_columns = ['num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 
                     'number_emergency', 'number_inpatient', 'number_diagnoses', 'number_of_medication_changes', 'number_of_medications', 'service_utilization']

# Apply log transformation if skewness is outside the acceptable range
for column in numerical_columns:
    skewness = calculate_skewness(data, column)
    if abs(skewness) > 2:
        data[column] = np.log1p(data[column])

# Encode and standardize numerical data
for column in numerical_columns:
    data[column] = (data[column] - data[column].mean()) / data[column].std()

data['readmitted'] = data['readmitted'].replace({'>30': 'NO', '<30': 'YES'})

data.drop(columns=['encounter_id', 'patient_nbr', 'A1Cresult', 'max_glu_serum'], inplace=True)

data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

output_file_path = 'preproc.csv'
data.to_csv(output_file_path, index=False)