import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Define the XOR input and output
X = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
y = [0, 0, 0, 0, 0, 1, 0, 1]

# Experiment with different MLP configurations
# This is a base setup, adjust the parameters as needed based on experiment results
clf = MLPClassifier(hidden_layer_sizes=(4, 2),  # Experiment with layer sizes
                    random_state=1,             # For reproducibility
                    verbose=True,               # To see progress
                    alpha=0.01,                 # Regularization term
                    activation='tanh',          # Soft activation function
                    learning_rate_init=0.1,    # Initial learning rate, adjust this
                    learning_rate='adaptive',   # Adjust learning rate dynamically
                    max_iter=10000)             # Increase if needed to ensure convergence

# Fit data to the model
clf.fit(X, y)

# Make predictions to test the learning
ypred = clf.predict(X)  # Using the same inputs as training for simplicity

# Calculate and print accuracy
accuracy = accuracy_score(y, ypred)
print(f'Accuracy: {accuracy}')

# Output the number of iterations to check the learning speed
print(f'Iterations: {clf.n_iter_}')
