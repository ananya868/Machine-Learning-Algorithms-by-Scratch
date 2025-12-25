"""Neural Network (Multilayer Perceptron)"""
# Simple MLP with 1 hidden layer.
# Forward Propagation: Compute inputs -> Hidden Layer -> Output Layer.
# Backward Propagation: Compute gradients and update weights.
# Activation Function: Sigmoid.
# Loss Function: Mean Squared Error (for simplicity).

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Input -> Hidden
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        # Hidden -> Output
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        
        return self.final_output

    def backward(self, X, y, output):
        # Calculate Error
        error = y - output
        
        # Gradients for Output Layer
        d_output = error * self.sigmoid_derivative(output)
        
        error_hidden_layer = d_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(self.hidden_output)
        
        # Update Weights and Biases
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        
        self.weights_input_hidden += X.T.dot(d_hidden_layer) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate

    def fit(self, X, y, epochs=1000):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return self.forward(X)

if __name__ == "__main__":
    # Example (XOR Problem)
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    nn.fit(X, y, epochs=10000)

    output = nn.predict(X)
    print("Neural Network (MLP) from Scratch")
    print("Predicted Output:")
    print(output)
    print("Rounded Output:")
    print(np.round(output))
