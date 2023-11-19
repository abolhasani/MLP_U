import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def plot_distribution(data, column, title):
    distribution = data[column].value_counts()
    distribution.plot(kind='pie', autopct='%1.1f%%', startangle=140)
    plt.axis('equal') 
    plt.title(title)
    plt.show()

file_path = 'preproc.csv'  
data = pd.read_csv(file_path)
categorical_columns = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for column in categorical_columns:
    data[column] = le.fit_transform(data[column])
plot_distribution(data, 'readmitted', "Original 'readmitted' Distribution")
plot_distribution(data, 'time_in_hospital', "Original 'time_in_hospital' Distribution")

#pu1 and pu2