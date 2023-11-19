import pandas as pd
from VMetrics import *
from svm import *
from utils import *

def train_full_model(X, y, best_params):
    svm_model = SVM(C=best_params[0], gamma_0=best_params[1], a=best_params[2], epochs=epochs)
    if best_params[3] == 'sch1':
        train_errors, test_errors, objective_values = svm_model.fit_sch1(X, y, X, y)
    else:
        train_errors, test_errors, objective_values = svm_model.fit_sch2(X, y, X, y)
    y_pred = svm_model.predict(X)
    return metricsSV(y, y_pred)
### Show best models first, that we know the params
file_path = 'smote_re_sv.csv' 
data = pd.read_csv(file_path)
X = data.drop(columns=['readmitted']).values
y = data['readmitted'].values
y = np.where(y == 0, -1, 1) 
X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.3)

print("\nprimal SVM with the first Learning Rate Schedule:\n")
svm_model = SVM(C=1, gamma_0=0.1, a=0.1, epochs=10)  
train_errors, test_errors, objective_values = svm_model.fit_sch1(X_train, y_train, X_test, y_test)
y_pred_test = svm_model.predict(X_test)
accuracy, r_squared, f1, auc, recall, precision = metricsSV(y_test, y_pred_test)
print(f'Accuracy: {accuracy}, R-squared: {r_squared}, F1 Score: {f1}, AUC: {auc}, Recall: {recall}, Precision: {precision}')
# Accuracy: 0.41739416581173416, R-squared: 0.41294144672869526, F1 Score: 0.2834924140832478, AUC: 0.853255339529522, Recall: 0.27797759471967615, Precision: 0.28923048032943005

print("\n primal SVM with the second Learning Rate Schedule:\n")
svm_model = SVM(C=0.1, gamma_0=1, epochs=10)  
train_errors, test_errors, objective_values = svm_model.fit_sch2(X_train, y_train, X_test, y_test)
y_pred_test = svm_model.predict(X_test)
accuracy, r_squared, f1, auc, recall, precision = metricsSV(y_test, y_pred_test)
print(f'Accuracy: {accuracy}, R-squared: {r_squared}, F1 Score: {f1}, AUC: {auc}, Recall: {recall}, Precision: {precision}')
# Accuracy: 0.3061497067420298, R-squared: 0.18809074260051406, F1 Score: 0.25034427085806543, AUC: 0.7972132496890488, Recall: 0.20389062922999576, Precision: 0.3242113031178894

# Now do 5-fold validation
file_path = 'smote_re_sv.csv'
data = pd.read_csv(file_path)
X = data.drop(columns=['readmitted']).values
y = data['readmitted'].values
y = np.where(y == 0, -1, 1)
results = []

####### Define hyperparameters
C_values = [0.1, 0.5, 1]
gamma_0_values = [0.01, 0.1, 1]
a_values = [0.01, 0.1, 1]
epochs = 10
#######

for C in C_values:
    for gamma_0 in gamma_0_values:
        for a in a_values:
            for schedule in ['sch1', 'sch2']:
                accuracies = []
                print(f"C: {C}, gamma: {gamma_0}, a: {a}, Schedule: {schedule}")
                for X_train, X_test, y_train, y_test in manual_k_fold_split(X, y):
                    svm_model = SVM(C=C, gamma_0=gamma_0, a=a, epochs=epochs)
                    if schedule == 'sch1':
                        train_errors, test_errors, objective_values = svm_model.fit_sch1(X_train, y_train, X_test, y_test)
                    else:
                        train_errors, test_errors, objective_values = svm_model.fit_sch2(X_train, y_train, X_test, y_test)
                    y_pred_test = svm_model.predict(X_test)
                    accuracy, _, _, _, _, _ = metricsSV(y_test, y_pred_test)
                    accuracies.append(accuracy)
                avg_accuracy = sum(accuracies) / len(accuracies)
                results.append((C, gamma_0, a, schedule, avg_accuracy))
best_sch1 = max((r for r in results if r[3] == 'sch1'), key=lambda x: x[4])
best_sch2 = max((r for r in results if r[3] == 'sch2'), key=lambda x: x[4])
#print("\nBest Parameters for Schedule 1 (sch1):", best_sch1)
#print("\nBest Parameters for Schedule 2 (sch2):", best_sch2)

print("\nBest Parameters for Schedule 1 (sch1):", best_sch1)
metrics_sch1 = train_full_model(X, y, best_sch1)
print("Full Model Metrics with Best Parameters for Schedule 1 (sch1):", metrics_sch1)
print("\nBest Parameters for Schedule 2 (sch2):", best_sch2)
metrics_sch2 = train_full_model(X, y, best_sch2)
print("Full Model Metrics with Best Parameters for Schedule 2 (sch2):", metrics_sch2)
"""

# Best Parameters for Schedule 1 (sch1): (1, 0.1, 0.1, 'sch1', 0.4167363530778164)
# Accuracy: 0.4136389152358646, R-squared: 0.3542504672545559, F1 Score: 0.27890677003798114, AUC: 0.8385805600984049, Recall: 0.2753701627556346, Precision: 0.2825354012137559
# Best Parameters for Schedule 2 (sch2): (0.1, 1, 0.1, 'sch2', 0.43515679442508703)
# Full Model Metrics with Best Parameters for Schedule 2 (sch2): (0.3662051984855172, 0.3889805114863766, 0.27580928388517495, 0.8472451278715942, 0.2441367989903448, 0.3169247315070283)
# Accuracy: 0.48845357232728753, R-squared: -0.5759990074594084, F1 Score: 0.237534476104412, AUC: 0.6043337299095227, Recall: 0.32416114279841735, Precision: 0.1874433599263122
"""

# We do not run the gaussian dual format, due to the method of implementation that requires gigabytes upon gigabytes of memory
"""
print("\n Guassian Dual SVM:\n")
C = 1.0  
gamma = 0.01  
alphas_sv, w, b, support_vectors, support_vector_labels = train_svm(X_train, y_train, C, gamma)
y_pred_test = predict(X_test, alphas_sv, support_vectors, support_vector_labels, b, gamma)
accuracy, r_squared, f1, auc, recall, precision = metricsSV(y_test, y_pred_test)
print(f'Accuracy: {accuracy}, R-squared: {r_squared}, F1 Score: {f1}, AUC: {auc}, Recall: {recall}, Precision: {precision}')
# numpy.core._exceptions._ArrayMemoryError: Unable to allocate 920. GiB for an array with shape (123513902803,) and data type float64
"""