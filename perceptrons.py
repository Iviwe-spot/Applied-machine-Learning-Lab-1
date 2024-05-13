import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the CSV file into a DataFrame
df = pd.read_csv('Percept1.csv', header=None, delimiter=';')

# Separate features and target variable
X = df.drop(columns=[2])
y = df[2]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Perceptron model
perceptron = Perceptron(random_state=42)

# Train the Perceptron model
perceptron.fit(X_train_scaled, y_train)

# Get the number of inputs
num_inputs = X_train_scaled.shape[1]

# Get the number of outputs
num_outputs = 1  # Perceptron has a single output

# Print the number of inputs and outputs
print("Number of inputs:", num_inputs)
print("Number of outputs:", num_outputs)

# Print the weights and threshold
print("Weights:", perceptron.coef_)
print("Threshold:", perceptron.intercept_)

# Predict on the testing set
y_pred = perceptron.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Define the decision boundary function
def plot_decision_boundary(X, y, model):
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

    # Plot the decision boundary
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                         np.linspace(ylim[0], ylim[1], 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.Paired)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plot_decision_boundary(X_train_scaled, y_train, perceptron)
plt.show()

# Inform the user about the expected input format
print("Please provide 5 input samples with features separated by semicolons.")
print("For example: 1.2;3.4")

# Ask the user for input and make predictions
for i in range(5):
    user_input = input("Input sample {}: ".format(i+1))
    features = np.array([list(map(float, user_input.split(';')))])
    features_scaled = scaler.transform(features)
    prediction = perceptron.predict(features_scaled)
    print("Output classification for input sample {}: {}".format(i+1, prediction))
