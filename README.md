Practciall 1
Write a Python program to plot a few activation functions that are being used inneural networks.

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
plt.plot(x, 1 / (1 + np.exp(-x)), label='Sigmoid')
plt.plot(x, np.tanh(x), label='tanh')
plt.plot(x, np.maximum(0, x), label='ReLU')
plt.plot(x, x, label='Identity')
plt.plot(x, np.exp(x) / np.sum(np.exp(x)), label='Softmax')

plt.xlabel('Input')
plt.ylabel('Activation')
plt.title('Activation Functions')
plt.legend()
plt.grid(True)
plt.show()

-------------------------------------------------------------------------

Pra-2
Generate ANDNOT function using McCulloch-Pitts neural net by a python  program

import numpy as np
def mp_neuron(inputs, weights, threshold):
    weighted_sum = np.dot(inputs, weights)
    output = 1 if weighted_sum >= threshold else 0
    return output
def and_not(x1, x2):
    weights = [1, -1] 
    threshold = 1   
    inputs = np.array([x1, x2])
    output = mp_neuron(inputs, weights, threshold)
    return output
print(and_not(0, 0)) 
print(and_not(1, 0))  
print(and_not(0, 1))  
print(and_not(1, 1))

-------------------------------------------------------------------------

par-3  Write a Python Program using Perceptron Neural Network to recognise even and odd numbers. 
Given numbers are in ASCII form 0 to 9 

import numpy as np

j = int(input("Enter a Number (0-9): "))
step_function = lambda x: 1 if x >= 0 else 0

training_data = [
    {'input': [1, 1, 0, 0, 0, 0], 'label': 1},
    {'input': [1, 1, 0, 0, 0, 1], 'label': 0},
    {'input': [1, 1, 0, 0, 1, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 1, 1], 'label': 0},
    {'input': [1, 1, 0, 1, 0, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 0, 1], 'label': 0},
    {'input': [1, 1, 0, 1, 1, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 1, 1], 'label': 0},
    {'input': [1, 1, 1, 0, 0, 0], 'label': 1},
    {'input': [1, 1, 1, 0, 0, 1], 'label': 0},
]

weights = np.array([0, 0, 0, 0, 0, 1])

for data in training_data:
    input = np.array(data['input'])
    label = data['label']
    output = step_function(np.dot(input, weights))
    error = label - output
    weights += input * error

input = np.array([int(x) for x in list('{0:06b}'.format(j))])
output = "odd" if step_function(np.dot(input, weights)) == 0 else "even"
print(j, " is ", output)

-------------------------------------------------------------------------

4. With a suitable example demonstrate the perceptron learning law with its decision regions using 
python. Give the output in graphical form.
# Import necessary libraries
import numpy as np   # For numerical operations like array, dot product, etc.
import matplotlib.pyplot as plt   # For plotting graphs

# Define the perceptron function
def perceptron(x, w, b):
    # Calculate the weighted sum and apply sign function to decide output (+1 or -1)
    return np.sign(np.dot(x, w) + b)

# Define the perceptron learning algorithm
def perceptron_learning(X, Y, eta, epochs):
    w = np.zeros(2)   # Initialize weights with zeros (2 weights because input has 2 features)
    b = 0             # Initialize bias with zero

    # Loop over the dataset multiple times (epochs)
    for epoch in range(epochs):
        # Loop over each training sample
        for i in range(X.shape[0]):
            y_pred = perceptron(X[i], w, b)   # Predict output using current weights and bias
            
            if y_pred != Y[i]:   # If prediction is wrong
                w += eta * Y[i] * X[i]  # Update the weights
                b += eta * Y[i]         # Update the bias

    return w, b  # After training, return the final weights and bias

# Training Data
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # Input features
Y = np.array([-1, -1, -1, 1])  # Correct labels for each input

# Train the perceptron
w, b = perceptron_learning(X, Y, eta=1, epochs=10)

# Prepare the plotting area
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # Minimum and maximum values for X-axis
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Minimum and maximum values for Y-axis

# Create a mesh grid for decision boundary plotting
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict output for each point in the mesh grid
Z = np.array([perceptron(np.array([x, y]), w, b) for x, y in np.c_[xx.ravel(), yy.ravel()]])
Z = Z.reshape(xx.shape)  # Reshape to match the grid shape

# Plot the decision boundary using contour plot
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot the original training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

# Add labels and titles to the graph
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Decision Regions')

# Set the limits for X and Y axes
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

# Display the plot
plt.show()

-------------------------------------------------------------------------

5. Write a python Program for Bidirectional Associative Memory with two pairs of vectors. 
import numpy as np
X = np.array([[1, 1, 1, -1], [-1, -1, 1, 1]])
Y = np.array([[1, -1], [-1, 1]])
W = np.dot(Y.T, X)

def bam(x):
    y = np.sign(np.dot(W, x))
    return y

x_test = np.array([1, -1, -1, -1])
y_test = bam(x_test)


print("Input x:", x_test)
print("Output y:", y_test)
-------------------------------------------------------------------------
6 Implement Artificial Neural Network training
process in Python by Using Forward
Propagation, Back Propagation.

import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.rand(2, 1)
        self.bias = np.random.rand(1)

    def train(self, X, y, epochs):
        for i in range(epochs):
            output = self.predict(X)
            error = y - output
            #backward_prop
            delta = error * output * (1 - output)
            self.weights += np.dot(X.T, delta)
            self.bias += np.sum(delta)

    def predict(self, X):
        return 1 / (1 + np.exp(-(np.dot(X, self.weights) + self.bias))) #forward_Prop

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

nn = NeuralNetwork()
nn.train(X, y, epochs=1000)

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = nn.predict(test_data)

for x, prediction in zip(test_data, predictions):
    print(f"Input: {x}, Prediction: {prediction}")


    -------------------------------------------------------------------------
7 Write a python program to show Back
Propagation Network for XOR function with
Binary Input and Output

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

on
def sigmoid_derivative(x):
    return x * (1 - x)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


np.random.seed(42)

weights_0 = 2 * np.random.random((2, 4)) - 1
weights_1 = 2 * np.random.random((4, 1)) - 1

for i in range(10000):
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, weights_0))
    layer_2 = sigmoid(np.dot(layer_1, weights_1))
    
    error = y - layer_2
    
    delta_2 = error * sigmoid_derivative(layer_2)
    delta_1 = delta_2.dot(weights_1.T) * sigmoid_derivative(layer_1)
    

    weights_1 += layer_1.T.dot(delta_2)
    weights_0 += layer_0.T.dot(delta_1)

output = sigmoid(np.dot(sigmoid(np.dot(X, weights_0)), weights_1))
print("Predicted Output:")
print(output)

-------------------------------------------------------------------------
8 Write a python program to design a Hopfield
Network which stores 4 vectors.
import numpy as np

class HopfieldNetwork:
    def __init__(self, n):
        self.n = n
        self.weights = np.zeros((n, n))

    def train(self, patterns):
        for pattern in patterns:
            pattern = np.reshape(pattern, (self.n, 1))
            self.weights += np.dot(pattern, pattern.T)
        np.fill_diagonal(self.weights, 0)

    def predict(self, pattern, max_iter=100):
        pattern = np.reshape(pattern, (self.n, 1))
        for _ in range(max_iter):
            new_pattern = np.sign(np.dot(self.weights, pattern))
            if np.array_equal(pattern, new_pattern):
                return np.squeeze(new_pattern)
            pattern = new_pattern
        return np.squeeze(pattern)

network = HopfieldNetwork(4)
patterns = [[1, -1, 1, -1], [-1, 1, -1, 1],[1,-1,-1,1],[-1,1,1,1]]
network.train(patterns)
corrupted_pattern = [1, -1, -1, -1]
predicted_pattern = network.predict(corrupted_pattern)
print("Corrupted pattern:", corrupted_pattern)
print("Predicted pattern:", predicted_pattern)

-------------------------------------------------------------------------

9. How to Train a Neural Network with
TensorFlow/ PyTorch and evaluation of
logistic regression using TensorFlow.

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
df=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(df.data,df.target,test_size=0.20,random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
model=tf.keras.models.Sequential([tf.keras.layers.Dense(1,activation='sigmoid',input_shape=(X_train.shape[1],))])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5)
y_pred=model.predict(X_test)
test_loss,test_accuracy=model.evaluate(X_test,y_test)
print("accuracy is",test_accuracy)

-------------------------------------------------------------------------
10. TensorFlow/ PyTorch implementation of

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model

# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Define input shape and number of classes
input_shape = (28, 28, 1)  # Example input shape for MNIST dataset
num_classes = 10            # Example number of classes for MNIST dataset

# Create the model
model = create_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=True)
plt.show()
