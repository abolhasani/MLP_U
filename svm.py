import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# svm class for primal svm
class SVM:
    # initializing the SVM entity with parameters
    def __init__(self, C, gamma_0=0.01, a=0.01, epochs=100):
        self.epochs = epochs 
        self.C = C             
        self.gamma_0 = gamma_0  
        self.a = a             
        self.w = None           
        self.b = 0             

    # shuffle data for primal SVM algorithm
    def shuffle_data(self, X, y):
        idx = np.random.permutation(len(y))
        return X[idx], y[idx]

    # hinge loss for slack, according to slide 58 lecture svm_intro
    def hinge_loss(self, x, y):
        return max(0, 1 - y * (np.dot(self.w, x) + self.b))

    # primal svm objective function, according to slide 66 lecture svm_intro
    def objective_function(self, X, y):
        hinge_losses = [self.hinge_loss(x, y[i]) for i, x in enumerate(X)]
        return 0.5 * np.dot(self.w, self.w) + self.C * np.sum(hinge_losses)
    
    # svm train fit algorithm for the first schedule of learning rate
    def fit_sch1(self, X_train, y_train, X_test, y_test):
        n_samples, n_features = X_train.shape
        self.w = np.zeros(n_features)
        train_errors = []
        test_errors = []
        objective_values = []
        for epoch in range(self.epochs):
            X_train, y_train = self.shuffle_data(X_train, y_train)
            for i in range(n_samples):
                gamma_t = self.sch1(epoch)
                x_i, y_i = X_train[i], y_train[i]
                if y_i * (np.dot(self.w, x_i) + self.b) < 1:
                    # update weight and bias for those samples that are misclassified
                    self.w += gamma_t * (y_i * x_i - (2 * self.C * self.w / n_samples))
                    self.b += gamma_t * y_i * self.C
                else:
                    # update weight for those samples that are correctly classified due to regularization
                    self.w -= gamma_t * (2 * self.C * self.w / n_samples)
            train_error = 1 - self.score(X_train, y_train)
            test_error = 1 - self.score(X_test, y_test)
            train_errors.append(train_error)
            test_errors.append(test_error)
            objective_values.append(self.objective_function(X_train, y_train))
        return train_errors, test_errors, objective_values
    
    # schedule of learning rate = gamma_0/(1+t*gamma_0/a), where t is epoch and we have a and gamma_0 given
    def sch1(self, epoch):
        return self.gamma_0 / (1 + (self.gamma_0 / self.a) * epoch) 

    # schedule of learning rate = gamma_0/(1+t), where t is epoch and we have gamma_0 given
    def sch2(self, epoch):
        return self.gamma_0 / (1 + epoch)

    # svm train fit algorithm for the second schedule of learning rate
    def fit_sch2(self, X_train, y_train, X_test, y_test):
        n_samples, n_features = X_train.shape
        self.w = np.zeros(n_features)
        train_errors = []
        test_errors = []
        objective_values = []
        for epoch in range(self.epochs):
            X_train, y_train = self.shuffle_data(X_train, y_train)
            gamma_t = self.sch2(epoch)
            for i in range(n_samples):
                if y_train[i] * (np.dot(self.w, X_train[i]) + self.b) < 1:
                    self.w += gamma_t * ((y_train[i] * X_train[i]) - (2 * self.C * self.w / n_samples))
                    self.b += gamma_t * y_train[i]
                else:
                    self.w -= gamma_t * (2 * self.C * self.w / n_samples)
            train_errors.append(1 - self.score(X_train, y_train))
            test_errors.append(1 - self.score(X_test, y_test))
            objective_values.append(self.objective_function(X_train, y_train))
        return train_errors, test_errors, objective_values

    # return the sign of objective function
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    # function to retrieve the accuracy (1-error) that is the mean of the ones correctly predicted
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    # plotting and error printing function
    def plotting(self, train_error, test_error, C):
        print(f'Training error: {train_error}')
        print(f'Test error: {test_error}')
        plt.title(f'Objective Function Vlaues (SVM Primal) for C={C}')
        plt.xlabel('Epoch')
        plt.ylabel('Objective Function Value')
        plt.legend()
        plt.show()

# svm dual train function with optimization of the dual objective function 
def fit_svm_dual(X, y, C):
    n_samples = X.shape[0]
    initial_alpha = np.zeros(n_samples)
    bounds = [(0, C) for _ in range(n_samples)]
    constraints = {'type': 'eq', 'fun': constraint_eq, 'args': (y,)}
    result = minimize(fun=dual_objective,
                      x0=initial_alpha,
                      args=(X, y),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    # optimized Lagrange multipliers
    alphas = result.x
    # eecover weight
    w = np.sum((alphas * y)[:, np.newaxis] * X, axis=0)
    # we recover bias by using the boundaries of support vectors: (0 < alpha < C)
    support_mask = (alphas > 1e-4) & (alphas < C)
    support_vectors = X[support_mask]
    support_labels = y[support_mask]
    support_alphas = alphas[support_mask]
    b = np.mean(support_labels - np.dot(support_vectors, w))
    return w, b

# dual objectivefor non gauss, according to slide 37 of lecture svm-dual-kernel-tricks
def dual_objective(alpha, X, y):
    return 0.5 * np.dot(alpha * y, np.dot(X, X.T).dot(alpha * y)) - np.sum(alpha)

# equality constraints passed to the optimizer for dual svm - sum(a_iy_i)=0, 0<a<c
def constraint_eq(alpha, y):
    return np.dot(alpha, y)

# return the sign of objective function used for dual objective outside of the svm class 
def predict_svm(X, w, b):
    return np.sign(np.dot(X, w) + b)

# function to retrieve the error that is the mean of the ones not correctly predicted
def calculate_error(y_true, y_pred):
    return np.mean(y_true != y_pred)

# gaussian kernel according to slide 90 lecture svm-dual-kernel-tricks
def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / gamma)

# dual objective function rewrote for gaussian
def objective_function(alphas, X, y, gamma):
    m = X.shape[0]
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma)
    return 0.5 * np.sum(alphas @ K @ alphas) - np.sum(alphas)

# prediction for dual gaussian
def predict(X, alphas, support_vectors, support_vector_labels, b, gamma):
    y_pred = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        s = 0
        for alpha, sv, sv_label in zip(alphas, support_vectors, support_vector_labels):
            s += alpha * sv_label * gaussian_kernel(X[i], sv, gamma)
        y_pred[i] = s
    return np.sign(y_pred + b)

# function to retrieve the error that is the mean of the ones not correctly predicted for gussian dual svm
def compute_error(X, y, alphas, support_vectors, support_vector_labels, b, gamma):
    y_pred = predict(X, alphas, support_vectors, support_vector_labels, b, gamma)
    return np.mean(y_pred != y)

# formula is exp(-||x-z||^2/c) given by slide 91 lecture svm-dual-kernel-tricks, where ||x-z||^2 is euclidean distance
def gaussian_kernel_matrix(X, gamma):
    sq_dists = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    return np.exp(-sq_dists / gamma)

# Revised svm_dual_objective function with vectorized kernel computation
def svm_dual_objective(alphas, X, y, gamma):
    K = gaussian_kernel_matrix(X, gamma)
    return 0.5 * np.dot(alphas, np.dot(K, alphas)) - np.sum(alphas)

# Equality constraint (sum of alphas * y = 0)
def equality_constraint(alphas, y):
    return np.dot(alphas, y)

# gaussian dual svm train function
def train_svm(X_train, y_train, C, gamma):
    m, n = X_train.shape
    alphas = np.zeros(m)
    constraint = {'type': 'eq', 'fun': equality_constraint, 'args': (y_train,)}
    bounds = [(0, C) for _ in range(m)]
    result = minimize(svm_dual_objective, alphas, args=(X_train, y_train, gamma), 
                      method='SLSQP', bounds=bounds, constraints=constraint)
    alphas = result.x
    sv_indices = alphas > 1e-5
    support_vectors = X_train[sv_indices]
    support_vector_labels = y_train[sv_indices]
    alphas_sv = alphas[sv_indices]
    b = np.mean([y_k - np.sum(alphas_sv * support_vector_labels * 
                              np.array([gaussian_kernel(x_k, x_i, gamma) for x_i in support_vectors])) 
                 for (x_k, y_k) in zip(support_vectors, support_vector_labels)])
    w = np.sum((alphas_sv * support_vector_labels)[:, np.newaxis] * support_vectors, axis=0)
    return alphas_sv, w, b, support_vectors, support_vector_labels

# load data and convert label to -1 +1
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y = np.where(y == 0, -1, 1)
    return X, y