# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 23:44:04 2023

@author: Marlon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def identity(x):
    return x

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def split_shuffle(inputs, outputs, split_treino=0.6, split_validacao=0.2, split_teste=0.2):

    # Calcula o número de amostras para cada split
    n_x = len(inputs)
    n_x_train = int(n_x * split_treino)
    n_x_val = int(n_x * split_validacao)
    n_x_teste = n_x - n_x_train - n_x_val

    # Shuffle
    shuffle_indices = np.random.permutation(n_x)
    inputs = inputs[shuffle_indices]
    outputs = outputs[shuffle_indices]

    # Realiza o split
    X_train, y_train = inputs[:n_x_train], outputs[:n_x_train]
    X_val, y_val = inputs[n_x_train:n_x_train + n_x_val], outputs[n_x_train:n_x_train + n_x_val]
    X_test, y_test = inputs[n_x_train + n_x_val:], outputs[n_x_train + n_x_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test

class MLP:
    def __init__(self, input_size, hidden_size, output_size, hidden_activation='sigmoid'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases with random values
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        
        self.original_w_b = [self.W1, self.b1, self.W2, self.b2]
        
        # Set the activation function for the hidden layer
        if hidden_activation == 'sigmoid':
            self.hidden_activation = sigmoid
            self.hidden_activation_derivative = sigmoid_derivative
        elif hidden_activation == 'relu':
            self.hidden_activation = relu
            self.hidden_activation_derivative = relu_derivative
        elif hidden_activation == 'tanh':
            self.hidden_activation = tanh
            self.hidden_activation_derivative = tanh_derivative
        else:
            raise ValueError("Invalid activation function")
        
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.hidden_activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = identity(self.z2)
        # self.a2 = relu(self.z2)
        
        return self.a2
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]  # n training examples
                
        delta2 = self.a2 - y
        dW2 = np.dot(self.a1.T, delta2) / m 
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        delta1 = np.dot(delta2, self.W2.T) * self.hidden_activation_derivative(self.a1) 
        dW1 = np.dot(X.T, delta1) / m  
        db1 = np.sum(delta1, axis=0, keepdims=True) / m 
        
        # Update
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, num_epochs, learning_rate):
        train_losses = []
        val_losses = []
        patience = 10
        best_val_loss = np.inf        
        num_no_improvement = 0
        
        for epoch in range(num_epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            # Compute loss
            loss = np.mean(np.square(output - y))
            train_losses.append(loss)            
            
            # early stopping based on validation loss
            val_output = self.forward(X_val)
            val_loss = np.mean(np.square(val_output - y_val))
            val_losses.append(val_loss)
            
            # Print loss a cada 100 épocas
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                num_no_improvement = 0
            else:
                num_no_improvement += 1
                if num_no_improvement >= patience:
                    print(f"Early stop at epoch {epoch+1}")
                    break
        
        return train_losses, val_losses
    
    def predict(self, X):
        return np.round(self.forward(X))

    # Fazer a predição é diferente aqui, não preciso arredonda a saída gerada.
    def predict_regressao(self, X):
        return self.forward(X)

def normalizar(x):
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

# Read the data from the Excel file
data = pd.read_excel('dadosmamografia.xlsx', header=None)

# # Separa input e labels
X = data.iloc[:, :5].values
y = data.iloc[:, 5].values.reshape(-1, 1)

# Train-validation-test 
train_ratio = 0.6  # 60% treino
val_ratio = 0.2  # 20% validacao
test_ratio = 0.2  # 20% teste

# Calculate the number of samples for each split
num_samples = len(X)
num_train = int(num_samples * train_ratio)
num_val = int(num_samples * val_ratio)
num_test = num_samples - num_train - num_val

# Shuffle the data
shuffle_indices = np.random.permutation(num_samples)
X = X[shuffle_indices]
y = y[shuffle_indices]

# Split the data into train, validation, and test sets
X_train, y_train = X[:num_train], y[:num_train]
X_val, y_val = X[num_train:num_train + num_val], y[num_train:num_train + num_val]
X_test, y_test = X[num_train + num_val:], y[num_train + num_val:]

# Normaliza
X_train = normalizar(X_train)
X_val = normalizar(X_val)
X_test = normalizar(X_test)

# Create and train the MLP
mlp = MLP(input_size=5, hidden_size=5, output_size=1, hidden_activation='tanh')
train_losses, val_losses = mlp.train(X_train, y_train, num_epochs=10000, learning_rate=0.01)

# Evaluate the model on the testing set
predictions = mlp.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Testing Accuracy: {accuracy * 100:.2f}%")

# Plot the training and validation losses
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Treino Loss')
plt.plot(val_losses, label='Validação Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Treino e Validação Loss')
plt.legend()
plt.show()



