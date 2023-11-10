import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
readmitted_data = pd.read_csv('smote_re.csv')
time_in_hospital_data = pd.read_csv('smote_time.csv')

# Create a pie chart for 'readmitted' labels
readmitted_counts = readmitted_data['readmitted'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(readmitted_counts, labels=readmitted_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Readmitted Labels')
plt.show()

# Create a pie chart for 'time_in_hospital' labels
time_in_hospital_counts = time_in_hospital_data['time_in_hospital'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(time_in_hospital_counts, labels=time_in_hospital_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Time in Hospital Labels')
plt.show()

