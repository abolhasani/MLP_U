import numpy as np

# error calculation for standard implementation - done in the main code 
def calculate_error(predictions, labels):
    misclassified = predictions != labels
    return np.mean(misclassified)

# According to slide 79 of the Perceptron lecture:
class Perceptron:
    # step 1, initialize weights to zero; weights[0] is the bias term, num_features + 1 adds one because of the biases.
    def __init__(self, num_features):
        self.weights = np.zeros(num_features + 1) 

    # calculate the weighted sum and apply the sign function for prediction
    def predict(self, features):
        # adding 1 for the bias in the features
        x = np.insert(features, 0, 1) 
        activation = np.dot(self.weights, x)
        return 1 if activation >= 0 else -1
    
    # step 2        
    def train(self, features, labels, T=10, learning_rate=1.0, seed=None):
        if seed is not None:
            np.random.seed(seed) 
        for epoch in range(T):
            # shuffle the data each epoch
            combined = np.c_[features, labels]
            np.random.shuffle(combined)
            shuffled_features = combined[:, :-1]
            shuffled_labels = combined[:, -1]
            # for each example, if y_i W^T.X <0, then update
            for x, y in zip(shuffled_features, shuffled_labels):
                prediction = self.predict(x)
                if y * prediction <= 0:  
                    self.update_weights(x, y, learning_rate)
    
    # the update value
    def update_weights(self, features, label, learning_rate):
        x = np.insert(features, 0, 1)  
        self.weights += learning_rate * label * x        
    
# According to slide 86 of the Perceptron lecture:
class VotedPerceptron:
    # step 1, initializations
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = []  
        self.counts = []  

    # step 2
    def train(self, training_inputs, training_labels, T=10, learning_rate=1):
        w = np.zeros(self.num_features)  # initialize weight vector to zero
        c = 1  # initialize C_m, the number of predictions made by w_m
        m = 0  # initialize m, the indice
        for epoch in range(T):
            # for each training example
            for x_i, y_i in zip(training_inputs, training_labels):
                # if y_i W^T.X <0, then update, update w_{m+1}
                if y_i * np.dot(w, x_i) <= 0:
                    self.weights.append(w)
                    self.counts.append(c)
                    w = w + learning_rate*y_i * x_i
                    c = 1  
                else:
                    c += 1  
        self.weights.append(w)
        self.counts.append(c)
        # returning the (w_1,c_1), ..., (w_k,c_k)
        return self.weights, self.counts

    # sgn(sum(c_i*sgn(W_i^T.X)))
    def predict(self, test_inputs):
        predictions = np.zeros(len(test_inputs))
        for i, x in enumerate(test_inputs):
            votes = sum(c * np.sign(np.dot(w, x)) for w, c in zip(self.weights, self.counts))
            predictions[i] = np.sign(votes)
        return predictions

# According to slide 89 of the Perceptron lecture:
class AveragedPerceptron:
    # step 1, initializing weights to zero and average to zero
    def __init__(self, num_features):
        self.w = np.zeros(num_features)
        self.a = np.zeros(num_features) 

    # step 2
    def train(self, features, labels, T=10, learning_rate=1):
        for _ in range(T):
            for xi, yi in zip(features, labels):
                # for each example, if y_i W^T.X <0, then update, update w
                if yi * np.dot(self.w, xi) <= 0:
                    self.w += learning_rate * yi * xi
                # a <- a+w
                self.a += self.w
        # Averaging the weight over all updates
        self.a /= (features.shape[0] * T)
        return self.a

    # sgn(sum(c_i*W_i^T.X))
    def predict(self, features):
        return np.sign(np.dot(features, self.a))

    # error return
    def test(self, features, labels):
        predictions = self.predict(features)
        errors = predictions != labels
        return np.mean(errors)