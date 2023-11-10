import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv('Data/concrete/concrete/train.csv', header=None)
test_data = pd.read_csv('Data/concrete/concrete/test.csv', header=None)
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.reshape(-1, 1)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values.reshape(-1, 1)
# Adding a bias term to the data
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

def compute_cost(X, y, weights):
    m = len(y)
    predictions = X.dot(weights)
    cost = (1/(2*m)) * np.sum(np.square(predictions-y))
    #cost = (1/(2)) * np.sum(np.square(predictions-y))
    return cost

def batch_gradient_descent(X, y, learning_rate=0.01, tolerance=1e-6, max_iterations=1000):
    m, n = X.shape
    weights = np.zeros((n, 1))
    costs = []
    for iteration in range(max_iterations):
        gradient = (1/m) * X.T.dot(X.dot(weights) - y)
        prev_weights = weights.copy()
        weights -= learning_rate * gradient
        if iteration % 100 == 0:
            learning_rate /= 2
        # Convergence check
        weight_difference_norm = np.linalg.norm(weights - prev_weights)
        if weight_difference_norm < tolerance:
            print("CONVERGED!")
            break
        cost_i=compute_cost(X, y, weights)
        costs.append(cost_i)
    return weights, costs, learning_rate

weights, costs, lr = batch_gradient_descent(X_train, y_train, learning_rate=0.5)
print("Weights: ", weights, "\nCost: ", costs[-1], "\nLearning Rate: ", lr)
plt.figure(figsize=(10, 6))
plt.plot(costs)
plt.title(f"Learning Rate = {lr}")
plt.xlabel("Iterations")
plt.ylabel("Cost Function")
plt.grid(True)
plt.show()

m = len(y_test)
predictions = X_test.dot(weights)
cost = (1/(2)) * np.sum(np.square(predictions-y_test))
print("Cost of the test data with the latest weight: ", cost)