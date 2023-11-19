from sklearn.metrics import r2_score, f1_score, roc_auc_score, recall_score, precision_score, accuracy_score
import numpy as np

def metrics_report(tree, test_data):
    errors = 0
    actual = []
    predict = []
    default_prediction = test_data.iloc[:, -1].mode()[0]  # most common class
    for index, row in test_data.iterrows():
        current_node = tree
        predict_label = None
        while current_node.label is None:
            current_value = row.iloc[current_node.attribute]
            current_node = current_node.children.get(current_value, None)
            if current_node is None:
                predict_label = default_prediction 
                break
        if current_node: 
            predict_label = current_node.label
            if current_node.label != row.iloc[-1]:
                errors += 1
        predict.append(predict_label)
        actual.append(row.iloc[-1])
    error_rate = errors / len(test_data)

    predict = np.array(predict)
    actual = np.array(actual)

    r_squared = r2_score(actual, predict) if None not in predict else None
    f1 = f1_score(actual, predict, average='macro')
    recall = recall_score(actual, predict, average='macro')
    precision = precision_score(actual, predict, average='macro')
    if len(set(actual)) == 2:
        auc = roc_auc_score(actual, predict)
    else:
        auc = None  
    return 1-error_rate, r_squared, f1, auc, recall, precision, predict