import numpy as np
import json
from numpy import random
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import random

# to make a more modular code, we define a class for the nodes of the trees, and make the initializations, the same as the one for DecisionTree.py in the DecisionTree folder for previous HW
class nodes:
    def __init__(self, attribute=None, value=None, label=None, children=None):
        self.attribute = attribute
        self.value = value
        self.label = label
        if children == None:
            self.children = {}
        else :
            self.children = children

def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    return X_train, X_test, y_train, y_test

# the preprocessing function to handle the bank data, I had to take it here to prevent several repetitions. 
def data_bank(data, label_to_b=None):
    if label_to_b is None:
        label_to_b = {
            'job': {'admin.': 0, 'unknown': 1, 'unemployed': 2, 'management': 3, 'housemaid': 4, 'entrepreneur': 5, 'student': 6, 'blue-collar': 7, 'self-employed': 8, 'retired': 9, 'technician': 10, 'services': 11},
            'marital': {'married': 0, 'divorced': 1, 'single': 2},
            'education': {'unknown': 0, 'secondary': 1, 'primary': 2, 'tertiary': 3},
            'default': {'yes': 1, 'no': 0},
            'housing': {'yes': 1, 'no': 0},
            'loan': {'yes': 1, 'no': 0},
            'contact': {'unknown': 0, 'telephone': 1, 'cellular': 2},
            'month': {'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5, 'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11},
            'poutcome': {'unknown': 0, 'other': 1, 'failure': 2, 'success': 3},
            'label': {'yes': 1, 'no': 0}
        }
    for column, mapping in label_to_b.items():
        data[column] = data[column].map(mapping)
    for column in ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']:
        median_val = data[column].median()
        data[column] = (data[column] > median_val).astype(int)
    return data, label_to_b

# i had ths problem with "None" values in the prediction results, replace a none 0,1 placeholder (-1)
def replace_none(x):
    return [-1 if v is None else v for v in x]

# used for bagged_tree function. it uses my error function implemented in the previous HW that traverses a tree and both calculates error and makes predictions
# the error(tree, data)[1] is because at place [0] I hold the errors
def predict_tree(tree, data):
    predictions = error(tree, data)[1]
    return np.array(replace_none(predictions))

# the majority vote mechanism used by bagging tree, and a basis for the random forest algorithm. 
def predict_bagged_trees(trees, data):
    all_predictions = [predict_tree(tree, data) for tree in trees]
    all_predictions = np.array(all_predictions).T
    majority_vote = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=1, arr=all_predictions)
    return majority_vote

# implementing the bagged_trees algorithm here. looping on tree numbers, using ID3 from last time to build a tree, appending them to make the bagged trees, 
# using the aforementioned error function to predict test and train instances and then do majority vote
def bagged_trees(train, test, max_trees):
    trees = []
    train_errors = []
    test_errors = []
    for n_trees in range(1, max_trees+1):
        # bootstrap the data
        bootstrap = train.sample(n=len(train), replace=True)
        # generate the trees from the existing ID3 algorithm with no defined depth
        tree = ID3(bootstrap.values, list(range(bootstrap.shape[1] - 1)), list(range(2)))
        trees.append(tree)
        # getting the predictions (built in the error function)
        train_predictions = [error(tree, train)[1] for tree in trees]
        train_predictions = np.array(train_predictions).T
        train_predictions = np.apply_along_axis(replace_none, axis=1, arr=train_predictions)
        #print(train_predictions)
        majority_vote_train = np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[0][np.argmax(np.unique(x, return_counts=True)[1])], axis=1, arr=train_predictions)
        train_error = np.mean(majority_vote_train != train.iloc[:, -1].values)
        train_errors.append(train_error)
        test_predictions = [error(tree, test)[1] for tree in trees]
        test_predictions = np.array(test_predictions).T
        test_predictions = np.apply_along_axis(replace_none, axis=1, arr=test_predictions)
        majority_vote_test = majority_vote_train = np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[0][np.argmax(np.unique(x, return_counts=True)[1])], axis=1, arr=test_predictions)
        test_error = np.mean(majority_vote_test != test.iloc[:, -1].values)
        test_errors.append(test_error)
        print(f"Number of Trees: {n_trees}, Train Error: {train_error}, Test Error: {test_error}")
    return train_errors, test_errors, trees

def bagged_trees_data(train, test, max_trees, end_label, heuristic):
    # Set_of_examples, attributes, labels, max_features=None, heuristic='HS'
    # tree = ID3(train.values, attribute_indices, end_label, heuristic='HS')
    # attribute_indices = [i for i in range(len(X_train.columns))]
    trees = []
    train_errors = []
    test_errors = []
    for n_trees in range(1, max_trees+1):
        # bootstrap the data
        bootstrap = train.sample(n=len(train), replace=True)
        # generate the trees from the existing ID3 algorithm with no defined depth
        tree = ID3(bootstrap.values, list(range(bootstrap.shape[1] - 1)), end_label, heuristic)
        trees.append(tree)
        # getting the predictions (built in the error function)
        train_predictions = [error(tree, train)[1] for tree in trees]
        train_predictions = np.array(train_predictions).T
        train_predictions = np.apply_along_axis(replace_none, axis=1, arr=train_predictions)
        #print(train_predictions)
        majority_vote_train = np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[0][np.argmax(np.unique(x, return_counts=True)[1])], axis=1, arr=train_predictions)
        train_error = np.mean(majority_vote_train != train.iloc[:, -1].values)
        train_errors.append(train_error)
        test_predictions = [error(tree, test)[1] for tree in trees]
        test_predictions = np.array(test_predictions).T
        test_predictions = np.apply_along_axis(replace_none, axis=1, arr=test_predictions)
        majority_vote_test = majority_vote_train = np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[0][np.argmax(np.unique(x, return_counts=True)[1])], axis=1, arr=test_predictions)
        test_error = np.mean(majority_vote_test != test.iloc[:, -1].values)
        test_errors.append(test_error)
        #print(f"Number of Trees: {n_trees}, Train Error: {train_error}, Test Error: {test_error}")
    return train_errors[-1], test_errors[-1], trees[-1]

# used for bias variance decomposition. Here I seperate biases and variances and compute them
def calculate_bias_variance(predictions, true_labels):
    bias = np.mean((predictions - true_labels)**2)
    variance = np.var(predictions)
    return bias, variance

# the main function for part (c) of the problem that iterates over the required times and makes bagged trees. then, those trees are used for extracting bias, variance, and squared errors after
# getting their predicitions based on the error function
def bias_variance_decomposition(train, test, n_iterations, n_trees, sample_size):
    all_bias_single_tree = []
    all_variance_single_tree = []
    all_bias_bagging = []
    all_variance_bagging = []
    all_squared_error_single_tree = []
    all_squared_error_bagging = []
    for i in range(n_iterations):
        print("Iteration: ", i)
        sampled_train = train.sample(n=sample_size, replace=False)
        _, _, trees = bagged_trees(sampled_train, test, n_trees)
        first_tree = trees[0]
        # get predictions
        predictions_single_tree = predict_tree(first_tree, test)
        predictions_bagging = predict_bagged_trees(trees, test)
        true_labels = test.iloc[:, -1].values
        bias_single_tree, variance_single_tree = calculate_bias_variance(predictions_single_tree, true_labels)
        bias_bagging, variance_bagging = calculate_bias_variance(predictions_bagging, true_labels)
        # using the squared error formula
        squared_error_single_tree = np.sum((predictions_single_tree - true_labels)**2) / len(true_labels)
        squared_error_bagging = np.sum((predictions_bagging - true_labels)**2) / len(true_labels)
        all_bias_single_tree.append(bias_single_tree)
        all_variance_single_tree.append(variance_single_tree)
        all_bias_bagging.append(bias_bagging)
        all_variance_bagging.append(variance_bagging)
        all_squared_error_single_tree.append(squared_error_single_tree)
        all_squared_error_bagging.append(squared_error_bagging)
    avg_bias_single_tree = np.mean(all_bias_single_tree)
    avg_variance_single_tree = np.mean(all_variance_single_tree)
    avg_bias_bagging = np.mean(all_bias_bagging)
    avg_variance_bagging = np.mean(all_variance_bagging)
    avg_squared_error_single_tree = np.mean(all_squared_error_single_tree)
    avg_squared_error_bagging = np.mean(all_squared_error_bagging)
    return avg_bias_single_tree, avg_variance_single_tree, avg_bias_bagging, avg_variance_bagging, avg_squared_error_single_tree, avg_squared_error_bagging

# the function for parts (d) and (e) of the problem. it is a modified version of the previous bias_variance_decomposition function that runs random_forest and makes forests instead of bagged trees.
# It iterates over the required amounts of time, and gets the subset feature sizes from the main code to build the forests. the rest is the same as the original bias_variance_decomposition function.
def bias_variance_decomposition_rf(feature_subset_sizes, train, test, n_iterations, n_trees, sample_size, max_features):
    train_errors = {size: [] for size in feature_subset_sizes}
    test_errors = {size: [] for size in feature_subset_sizes}
    all_bias_single_tree = []
    all_variance_single_tree = []
    all_bias_bagging = []
    all_variance_bagging = []
    all_squared_error_single_tree = []
    all_squared_error_bagging = []
    for i in range(n_iterations):
        neg = i+1
        print("Iteration: ", neg)
        sampled_train = train.sample(n=sample_size, replace=False)
        train_errors_rf, test_errors_rf, trees = random_forest(sampled_train, test, n_trees, max_features)  # You might need to provide your bagged_trees function or use random_forest if that's what you want.
        train_errors[max_features] = train_errors_rf
        test_errors[max_features] = test_errors_rf
        first_tree = trees[0]
        predictions_single_tree = predict_tree(first_tree, test)
        predictions_bagging = predict_bagged_trees(trees, test)
        true_labels = test.iloc[:, -1].values
        squared_error_single_tree = np.sum((predictions_single_tree - true_labels)**2) / len(true_labels)
        squared_error_bagging = np.sum((predictions_bagging - true_labels)**2) / len(true_labels)
        all_squared_error_single_tree.append(squared_error_single_tree)
        all_squared_error_bagging.append(squared_error_bagging)
        bias_single_tree, variance_single_tree = calculate_bias_variance(predictions_single_tree, true_labels)
        bias_bagging, variance_bagging = calculate_bias_variance(predictions_bagging, true_labels)
        all_bias_single_tree.append(bias_single_tree)
        all_variance_single_tree.append(variance_single_tree)
        all_bias_bagging.append(bias_bagging)
        all_variance_bagging.append(variance_bagging)
    avg_bias_single_tree = np.mean(all_bias_single_tree)
    avg_variance_single_tree = np.mean(all_variance_single_tree)
    avg_bias_bagging = np.mean(all_bias_bagging)
    avg_variance_bagging = np.mean(all_variance_bagging)
    avg_squared_error_single_tree = np.mean(all_squared_error_single_tree)
    avg_squared_error_bagging = np.mean(all_squared_error_bagging)
    return avg_bias_single_tree, avg_variance_single_tree, avg_bias_bagging, avg_variance_bagging, train_errors, test_errors, avg_squared_error_single_tree, avg_squared_error_bagging



# the modified ID3 function that has been changed to accept custom randomly selected features. 
def ID3_rf(Set_of_examples, attributes, labels, max_features=None, heuristic='HS'):
    unique_labels = np.unique(Set_of_examples[:, -1])
    # base cases and termination constraints:
    if len(unique_labels) == 1:
        return nodes(label=unique_labels[0])
    if len(attributes) == 0 or (max_features and len(attributes) <= max_features):
        return nodes(label=np.bincount(Set_of_examples[:, -1]).argmax())
    # random feature selection for Random Forests. the main change against the original ID3
    if max_features:
        attributes = random.sample(attributes, min(len(attributes), max_features))
    # selecting the best attribute and splitting
    best_gain = 0
    best_attribute = None
    for attribute in attributes:
        current_gain = information_gain(Set_of_examples, attribute, labels, heuristic)
        if current_gain > best_gain:
            best_gain = current_gain
            best_attribute = attribute
    if best_gain == 0:
        return nodes(label=np.bincount(Set_of_examples[:, -1]).argmax())
    root = nodes(attribute=best_attribute)
    for value in np.unique(Set_of_examples[:, best_attribute]):
        subset_values = Set_of_examples[Set_of_examples[:, best_attribute] == value]
        if len(subset_values) == 0:
            root.children[value] = nodes(label=np.bincount(Set_of_examples[:, -1]).argmax())
        else:
            remaining_attributes = [attr for attr in attributes if attr != best_attribute]
            root.children[value] = ID3_rf(subset_values, remaining_attributes, labels, max_features, heuristic)
    return root

#sampled_train = train.sample(n=sample_size, replace=False)
#train_errors_rf, test_errors_rf, trees = random_forest(sampled_train, test, n_trees, max_features)
def random_forest_data(train, test, end_label, max_trees, max_features=None):
    trees = []
    train_errors = []
    test_errors = []
    for n_trees in range(1, max_trees+1):
        #train = train.sample(frac=1).reset_index(drop=True)
        #test = test.sample(frac=1).reset_index(drop=True)
        # bootstrap the data
        bootstrap = train.sample(n=len(train), replace=True)
        tree = ID3_rf(bootstrap.values, list(range(bootstrap.shape[1] - 1)), end_label, max_features, heuristic='HS')
        #tree = ID3_rf(bootstrap.values, list(range(bootstrap.shape[1] - 1)), list(range(2)), max_features, heuristic='HS')
        trees.append(tree)
        # getting the predictions (built in the error function)
        train_predictions = [error(tree, train)[1] for tree in trees]
        train_predictions = np.array(train_predictions).T
        train_predictions = np.apply_along_axis(replace_none, axis=1, arr=train_predictions)
        #print(train_predictions)
        majority_vote_train = np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[0][np.argmax(np.unique(x, return_counts=True)[1])], axis=1, arr=train_predictions)
        train_error = np.mean(majority_vote_train != train.iloc[:, -1].values)
        train_errors.append(train_error)
        test_predictions = [error(tree, test)[1] for tree in trees]
        test_predictions = np.array(test_predictions).T
        test_predictions = np.apply_along_axis(replace_none, axis=1, arr=test_predictions)
        #majority_vote_test = majority_vote_train = np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[0][np.argmax(np.unique(x, return_counts=True)[1])], axis=1, arr=test_predictions)
        majority_vote_test = np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[0][np.argmax(np.unique(x, return_counts=True)[1])], axis=1, arr=test_predictions)
        test_error = np.mean(majority_vote_test != test.iloc[:, -1].values)
        test_errors.append(test_error)
        print(f"Number of Trees: {n_trees}, Train Error: {train_error}, Test Error: {test_error}")
    return train_errors, test_errors, trees[-1]

# a modified function of bagged trees that has the job of iterating over max_trees size, bootstrapping the data, and build max_trees trees and append them to make the forest
def random_forest(train, test, end_label, max_trees, max_features=None):
    trees = []
    train_errors = []
    test_errors = []
    for n_trees in range(1, max_trees+1):
        #train = train.sample(frac=1).reset_index(drop=True)
        #test = test.sample(frac=1).reset_index(drop=True)
        # bootstrap the data
        bootstrap = train.sample(n=len(train), replace=True)
        tree = ID3_rf(bootstrap.values, list(range(bootstrap.shape[1] - 1)), end_label, max_features, heuristic='HS')
        #tree = ID3_rf(bootstrap.values, list(range(bootstrap.shape[1] - 1)), list(range(2)), max_features, heuristic='HS')
        trees.append(tree)
        # getting the predictions (built in the error function)
        train_predictions = [error(tree, train)[1] for tree in trees]
        train_predictions = np.array(train_predictions).T
        train_predictions = np.apply_along_axis(replace_none, axis=1, arr=train_predictions)
        #print(train_predictions)
        majority_vote_train = np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[0][np.argmax(np.unique(x, return_counts=True)[1])], axis=1, arr=train_predictions)
        train_error = np.mean(majority_vote_train != train.iloc[:, -1].values)
        train_errors.append(train_error)
        test_predictions = [error(tree, test)[1] for tree in trees]
        test_predictions = np.array(test_predictions).T
        test_predictions = np.apply_along_axis(replace_none, axis=1, arr=test_predictions)
        #majority_vote_test = majority_vote_train = np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[0][np.argmax(np.unique(x, return_counts=True)[1])], axis=1, arr=test_predictions)
        majority_vote_test = np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[0][np.argmax(np.unique(x, return_counts=True)[1])], axis=1, arr=test_predictions)
        test_error = np.mean(majority_vote_test != test.iloc[:, -1].values)
        test_errors.append(test_error)
        print(f"Number of Trees: {n_trees}, Train Error: {train_error}, Test Error: {test_error}")
    return train_errors, test_errors, trees

import numpy as np

class Node:
    def __init__(self, label=None, attribute=None, children=None):
        self.label = label
        self.attribute = attribute
        self.children = {} if children is None else children

# modified ID3 from the previous DecisionTree.py for HW1 that determines no depth. it is not comfortable with the codes of the HW1, as they force a tree depyth
def ID3(Set_of_examples, attributes, labels, heuristic='HS'):
    unique_labels = np.unique(Set_of_examples[:, -1])
    if len(unique_labels) == 1:
        return nodes(label=unique_labels[0])
    if len(attributes) == 0:
        return nodes(label=np.bincount(Set_of_examples[:, -1]).argmax())
    best_gain = 0
    best_attribute = None
    for attribute in attributes:
        current_gain = information_gain(Set_of_examples, attribute, labels, heuristic)
        if current_gain > best_gain:
            best_gain = current_gain
            best_attribute = attribute
    if best_gain == 0:
        return nodes(label=np.bincount(Set_of_examples[:, -1]).argmax())
    root = nodes(attribute=best_attribute)
    for value in np.unique(Set_of_examples[:, best_attribute]):
        subset_values = Set_of_examples[Set_of_examples[:, best_attribute] == value]
        if len(subset_values) == 0:
            root.children[value] = nodes(label=np.bincount(Set_of_examples[:, -1]).argmax())
        else:
            root.children[value] = ID3(subset_values, [attr for attr in attributes if attr != best_attribute], labels, heuristic)
    return root

# computing entropy H(S) based on page 32 decision tree learning lecture
def HS(Set_of_examples, labels):
    # Count the unique occurrence of each label in S using bincount - only for integers
    label_count = np.bincount(Set_of_examples[:, -1], minlength=len(labels))
    # computing the positive and negative probabilities and getting their log 2
    p = label_count / (len(Set_of_examples) +1)
    # computing log probabilities from the p array
    logtwos = np.log2(p, where = (label_count / len(Set_of_examples)>0))
    entropy = -np.sum(p * logtwos)
    return entropy

def ME(Set_of_examples, labels):
    label_count = np.bincount(Set_of_examples[:, -1], minlength=len(labels))
    # find the most repeated label value
    max = np.max(label_count)
    MajErr = 1 - max / (len(Set_of_examples))
    return MajErr

# computing Gini index based on page 11 decision tree learning discussion
def gini_index(Set_of_examples, labels):
    label_count = np.bincount(Set_of_examples[:, -1], minlength=len(labels))
    p = (label_count / len(Set_of_examples)+1)
    # same as entropy till here. Then, we get the sum of their squares
    p_2 = 1 - np.sum(p**2)
    return p_2

# this part computes |Sv|/|S|, which will be used to compute gain
def SvOnS (Set_of_examples, attributes, value):
    subset_values = Set_of_examples[Set_of_examples[:, attributes] == value]
    svs = len(subset_values) / len(Set_of_examples)
    return svs, subset_values

# this part computes total gain, as in H(S) - SUM(W*H(subset_values)) based on the heuristics given
def information_gain(Set_of_examples, attributes, labels, heuristic):
    attribute_values = np.unique(Set_of_examples[:, attributes])
    gain = 0
    h_methods = {'gini': gini_index, 'ME': ME, 'HS': HS}
    if heuristic in h_methods:
        gain_function = h_methods[heuristic]
        gain = gain_function(Set_of_examples, labels)
        #This loop computes the ultimate gain from this heuristic method
        for value in attribute_values:
            w, subset_values = SvOnS (Set_of_examples, attributes, value)
            gain -= w * gain_function(subset_values, labels)
    return gain

# first crawls the tree with test data given to it to predict the label then checks the prediction 
def error(tree, test_data):
    errors = 0
    predict = []
    for index, row in test_data.iterrows():
        current_node = tree
        predict_labels = None
        # we have to navigate to reach a leaf node or a label that doesn't exist (which is an error)
        while current_node.label is None:
            current_value = row.iloc[current_node.attribute]
            current_node = current_node.children.get(current_value, None)
            if current_node is None:
                errors += 1
                break
        # After getting the label, we check whether it is equal to the correct label or not. 
        if current_node: 
            predict_labels = current_node.label
            if current_node.label != row.iloc[-1]:
                errors += 1
        predict.append(predict_labels)
    return errors / len(test_data), predict


# recursive function to print the tree. I used it for the test1 and 2 files that I knew the answer from part 1 of the HW1 to check them
def print_tree(node, depth=0, prefix="Root: "):
    # set indentation to make the results a bit more clear
    indent = "   " * depth
    # print the leaf value
    if node.label is not None:
        print(f"{indent}{prefix}Leaf: [Label: {node.label}]")
        return
    # if not leaf, then print the splitting attribute
    print(f"{indent}{prefix}[Attribute: {node.attribute}]")
    # for the splitted branches, print the children by exectuing the function on them
    for value, child_node in node.children.items():
        children_node = f"Value {value} -> "
        print_tree(child_node, depth + 1, prefix=children_node)
