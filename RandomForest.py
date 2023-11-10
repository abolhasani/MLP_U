from DecisionTree import *
import pandas as pd
import matplotlib.pyplot as plt

columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
train = pd.read_csv('Data/Bank/train.csv', names=columns)
train, train_with_labels = data_bank(train)
test = pd.read_csv('Data/Bank/test.csv', names=columns)
test, test_labels = data_bank(test, train_with_labels)
feature_subset_sizes = [2, 4, 6]
max_trees = 500 # 10 for testing and 500 for full run
n_iterations = 100 # 1 for testing and 100 for full run
n_trees = max_trees 
sample_size = 1000 
subsub = feature_subset_sizes
single_tree_predictions = {size: [] for size in feature_subset_sizes}
forest_predictions = {size: [] for size in feature_subset_sizes}
# a for loop for covering different subset sizes, in the loop we build the max_trees sized forests (it is called by the bias_variance_decomposition_rf, which is in the main DecisionTree library)
# and compute the bias, variance, and squared error for them and do the plotting as asked in the problem
for feature_subset_size in feature_subset_sizes:
    print(f"Working on feature subset size: {feature_subset_size}")
    results = bias_variance_decomposition_rf(subsub, train, test, n_iterations, n_trees, sample_size=sample_size, max_features=feature_subset_size)
    train_errors = results[4]
    test_errors = results[5]
    plt.plot(range(1, max_trees + 1), train_errors[feature_subset_size], label=f'Train, subset={feature_subset_size}')
    plt.plot(range(1, max_trees + 1), test_errors[feature_subset_size], label=f'Test, subset={feature_subset_size}')
    print(f"For feature subset size {feature_subset_size}:")
    print(f"Single Tree - Avg Bias: {results[0]}, Avg Variance: {results[1]}, Avg Squared Error: {results[6]}")
    print(f"Forest - Avg Bias: {results[2]}, Avg Variance: {results[3]}, Avg Squared Error: {results[7]}")
    print("======================================")
plt.xlabel('Random Forests')
plt.ylabel('Error Rate')
plt.legend()
plt.title('Error Rates for Different Subset Sizes and Number of Trees')
plt.show()
