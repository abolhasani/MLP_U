import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# the J function from slide 33 of lms regression
def compute_cost(X, y, weights):
    m = len(y)
    predictions = X.dot(weights)
    cost = (1/(2*m)) * np.sum(np.square(predictions-y))
    #cost = (1/(2)) * np.sum(np.square(predictions-y))
    return cost

# the cost gradient function from slide 33 of lms-regression
def gradient(X, y, weights):
    m = len(y)
    predictions = X.dot(weights)
    #print("X.T: ", X.T, "y - predictions: ", y - predictions)
    grad = -(1/m) * X.T.dot(y - predictions)
    #grad = X.T @ (X @ weights - y)
    #grad = -1 * X.T.dot(y - predictions)
    return grad

# the main SGD Algorithm
def stochastic_gradient_descent(X, y, learning_rate=0.1, max_iter=10000):
    m, n = X.shape
    weights = np.ones((X.shape[1], 1))
    costs = []
    for iteration in range(max_iter):
        i = np.random.randint(m)
        X_i = X[i:i+1]
        y_i = y[i:i+1]
        # calculate cost gradient
        grad = gradient(X_i, y_i, weights)
        #print(iteration, grad)
        # update weights and cost
        weights = weights - learning_rate * grad
        cost = compute_cost(X, y, weights)
        cost = compute_cost(X, y, weights)
        costs.append(cost)
        # halve the learning rate every 100 iterations
        if iteration % 100 == 0:
            learning_rate /= 2
        # check convergence criteria
        if iteration > 1 and abs(costs[-1] - costs[-2]) < 0.000001:
            break
    return weights, costs, learning_rate, iteration

train_data = pd.read_csv('Data/concrete/concrete/train.csv', header=None)
test_data = pd.read_csv('Data/concrete/concrete/test.csv', header=None)
# separating features and outputs
X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values.reshape(-1, 1)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values.reshape(-1, 1)
# adding a bias term to the data
X = np.hstack([np.ones((X.shape[0], 1)), X])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
# run the SGD algorithm and do the reportings: 
weights, costs, lr, it= stochastic_gradient_descent(X, y)
plt.plot(costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('SGD Convergence')
plt.show()
print("Converged Itteration: ", it)
print("Learning rate: ", lr)
print("Learned Weight Vector: ", weights)
print("Final Cost on Training Data: ", costs[-1])
m = len(y_test)
predictions = X_test.dot(weights)
#cost = (1/(2)) * np.sum(np.square(predictions-y_test))
print("Cost of the test data with the latest weight: ", costs[-1])

"""
# The analytical way
train_data = pd.read_csv('Data/concrete/concrete/train.csv', header=None)
test_data = pd.read_csv('Data/concrete/concrete/test.csv', header=None)
# Separating features and outputs
X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values.reshape(-1, 1)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values.reshape(-1, 1)
X = np.hstack([np.ones((X.shape[0], 1)), X])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
"""
# Using the provided formula in slide 49 of lms regression
def compute_optimal_weights(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
optimal_weights = compute_optimal_weights(X, y)
print("Optimal Weight Vector: ", optimal_weights)

m = len(y_test)
predictions = X_test.dot(optimal_weights)
cost = (1/(2)) * np.sum(np.square(predictions-y_test))
print("Cost of the test data with the optimal analytical weight: ", cost)