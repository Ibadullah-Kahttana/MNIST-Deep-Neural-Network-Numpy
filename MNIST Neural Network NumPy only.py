import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

"""**Load the Dataset**"""

# Load the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

"""**Flatten the input data to 1D array**"""

# Flatten the input data to 1D array
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

# Normalize the data to range [0, 1]
train_X = train_X / 255.0
test_X = test_X / 255.0

# Convert labels to one-hot vectors
train_y = np.eye(10)[train_y]
test_y = np.eye(10)[test_y]

"""**Activation Functions**"""

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the softmax activation function
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

"""**Neural Network Parameters**"""

# Define the neural network model
np.random.seed(42)
input_dim = train_X.shape[1]
print(f"input dim : {input_dim}")

hidden_dim = 16
output_dim = 10
learning_rate = 0.10

"""**Training simple hidden layer**"""

# Initialize weights and biases
W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
b2 = np.zeros(output_dim)


# Train the model
num_epochs = 50
batch_size = 128
num_batches = int(np.ceil(train_X.shape[0] / batch_size))
print(f"No of batches : {num_batches}")

loss_history_one_hidden = []
loss_history_one_hidden_momentum = []
loss_history_one_hidden_adam = []

test_accuracy_one_hidden = []
test_accuracy_one_hidden_momentum = []
test_accuracy_one_hidden_adam = []

for epoch in range(num_epochs):
    # Shuffle the data for each epoch
    perm = np.random.permutation(train_X.shape[0])
    train_X = train_X[perm]
    train_y = train_y[perm]
    
    # Mini-batch gradient descent
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, train_X.shape[0])
        X = train_X[start:end]
        y = train_y[start:end]
        
        # Forward pass
        h = sigmoid(np.dot(X, W1) + b1)
        y_pred = softmax(np.dot(h, W2) + b2)
        
        # Compute the cross-entropy loss
        loss = -np.mean(y * np.log(y_pred))
        loss_history_one_hidden.append(loss)
        
        # Backward pass
        grad_y_pred = (y_pred - y) / y.shape[0]
        grad_W2 = np.dot(h.T, grad_y_pred)
        grad_b2 = np.sum(grad_y_pred, axis=0)
        grad_h = np.dot(grad_y_pred, W2.T) * h * (1 - h)
        grad_W1 = np.dot(X.T, grad_h)
        grad_b1 = np.sum(grad_h, axis=0)
        
        # Update the weights and biases
        W2 -= learning_rate * grad_W2
        b2 -= learning_rate * grad_b2
        W1 -= learning_rate * grad_W1
        b1 -= learning_rate * grad_b1
        
    print("Epoch {}: loss = {:.6f}".format(epoch+1, loss))


# Evaluate the model on test data
h = sigmoid(np.dot(test_X, W1) + b1)
y_pred = softmax(np.dot(h, W2) + b2)
accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(test_y, axis=1))
test_accuracy_one_hidden.append(accuracy)
print("Test accuracy One Hidden Layer = {:.2f}%".format(accuracy * 100))

print("\n\n")

plt.plot(loss_history_one_hidden , color='green')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Learning Curve (One Hidden Layer)")
plt.show()

print("\n\n")

plt.plot(loss_history_one_hidden[::469] , color='green')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Learning Curve (One Hidden Layer)")
plt.show()

"""**Train the Neural Network with Momentum**"""

# Initialize weights and biases
W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
b2 = np.zeros(output_dim)

# Initialize the velocity for each parameter
v_W1 = np.zeros_like(W1)
v_b1 = np.zeros_like(b1)
v_W2 = np.zeros_like(W2)
v_b2 = np.zeros_like(b2)

# Train the model with momentum
momentum = 0.9
loss_history_one_hidden_momentum = []
for epoch in range(num_epochs):
    # Shuffle the data for each epoch
    perm = np.random.permutation(train_X.shape[0])
    train_X = train_X[perm]
    train_y = train_y[perm]
    
    # Mini-batch gradient descent with momentum
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, train_X.shape[0])
        X = train_X[start:end]
        y = train_y[start:end]
        
        # Forward pass
        h = sigmoid(np.dot(X, W1) + b1)
        y_pred = softmax(np.dot(h, W2) + b2)
        
        # Compute the cross-entropy loss
        loss = -np.mean(y * np.log(y_pred))
        loss_history_one_hidden_momentum.append(loss)
        
        # Backward pass
        grad_y_pred = (y_pred - y) / y.shape[0]
        grad_W2 = np.dot(h.T, grad_y_pred)
        grad_b2 = np.sum(grad_y_pred, axis=0)
        grad_h = np.dot(grad_y_pred, W2.T) * h * (1 - h)
        grad_W1 = np.dot(X.T, grad_h)
        grad_b1 = np.sum(grad_h, axis=0)
        
        # Update the velocity for each parameter
        v_W2 = momentum * v_W2 - learning_rate * grad_W2
        v_b2 = momentum * v_b2 - learning_rate * grad_b2
        v_W1 = momentum * v_W1 - learning_rate * grad_W1
        v_b1 = momentum * v_b1 - learning_rate * grad_b1
        
        # Update the weights and biases using the velocity
        W2 += v_W2
        b2 += v_b2
        W1 += v_W1
        b1 += v_b1
        
    print("Epoch {}: loss = {:.6f}".format(epoch+1, loss))

# Evaluate the model on test data
h = sigmoid(np.dot(test_X, W1) + b1)
y_pred = softmax(np.dot(h, W2) + b2)
accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(test_y, axis=1))
test_accuracy_one_hidden_momentum.append(accuracy)
print("Test accuracy one hidden momentum= {:.2f}%".format(accuracy * 100))

plt.plot(loss_history_one_hidden_momentum , color='blue')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve (One Hidden Layer with Momentum)")
plt.show()

print("\n\n")

plt.plot(loss_history_one_hidden_momentum[::469] , color='blue')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve (One Hidden Layer with Momentum)")
plt.show()

"""**Train the Neural Network with Momentum**"""

# Initialize weights and biases
# 0.01
W1 = np.random.randn(input_dim, hidden_dim)  / np.sqrt(input_dim)
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
b2 = np.zeros(output_dim)

# Initialize the first and second moments for each parameter
m_W1 = np.zeros_like(W1)
m_b1 = np.zeros_like(b1)
m_W2 = np.zeros_like(W2)
m_b2 = np.zeros_like(b2)
v_W1 = np.zeros_like(W1)
v_b1 = np.zeros_like(b1)
v_W2 = np.zeros_like(W2)
v_b2 = np.zeros_like(b2)
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Train the model with Adam optimizer
for epoch in range(num_epochs):
    # Shuffle the data for each epoch
    perm = np.random.permutation(train_X.shape[0])
    train_X = train_X[perm]
    train_y = train_y[perm]
    
    # Mini-batch gradient descent with Adam optimizer
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, train_X.shape[0])
        X = train_X[start:end]
        y = train_y[start:end]
        
        # Forward pass
        h = sigmoid(np.dot(X, W1) + b1)
        y_pred = softmax(np.dot(h, W2) + b2)
        
        # Compute the cross-entropy loss
        loss = -np.mean(y * np.log(y_pred))
        loss_history_one_hidden_adam.append(loss)
        
        # Backward pass
        grad_y_pred = (y_pred - y) / y.shape[0]
        grad_W2 = np.dot(h.T, grad_y_pred)
        grad_b2 = np.sum(grad_y_pred, axis=0)
        grad_h = np.dot(grad_y_pred, W2.T) * h * (1 - h)
        grad_W1 = np.dot(X.T, grad_h)
        grad_b1 = np.sum(grad_h, axis=0)
        
        # Update the first and second moments for each parameter
        m_W2 = beta1 * m_W2 + (1 - beta1) * grad_W2
        m_b2 = beta1 * m_b2 + (1 - beta1) * grad_b2
        m_W1 = beta1 * m_W1 + (1 - beta1) * grad_W1
        m_b1 = beta1 * m_b1 + (1 - beta1) * grad_b1
        v_W2 = beta2 * v_W2 + (1 - beta2) * grad_W2**2
        v_b2 = beta2 * v_b2 + (1 - beta2) * grad_b2**2
        v_W1 = beta2 * v_W1 + (1 - beta2) * grad_W1**2
        v_b1 = beta2 * v_b1 + (1 - beta2) * grad_b1**2
        
        # Compute the bias-corrected first and second moments
        m_W2_corrected = m_W2 / (1 - beta1**(epoch*num_batches+i+1))
        m_b2_corrected = m_b2 / (1 - beta1**(epoch*num_batches+i+1))
        m_W1_corrected = m_W1 / (1 - beta1**(epoch*num_batches+i+1))
        m_b1_corrected = m_b1 / (1 - beta1**(epoch*num_batches+i+1))
        v_W2_corrected = v_W2 / (1 - beta2**(epoch*num_batches+i+1))
        v_b2_corrected = v_b2 / (1 - beta2**(epoch*num_batches+i+1))
        v_W1_corrected = v_W1 / (1 - beta2**(epoch*num_batches+i+1))
        v_b1_corrected = v_b1 / (1 - beta2**(epoch*num_batches+i+1))

        # Update the parameters with Adam optimizer
        W2 -= learning_rate * m_W2_corrected / (np.sqrt(v_W2_corrected) + epsilon)
        b2 -= learning_rate * m_b2_corrected / (np.sqrt(v_b2_corrected) + epsilon)
        W1 -= learning_rate * m_W1_corrected / (np.sqrt(v_W1_corrected) + epsilon)
        b1 -= learning_rate * m_b1_corrected / (np.sqrt(v_b1_corrected) + epsilon)
  
    print("Epoch {}: loss = {:.6f}".format(epoch+1, loss))
    
# Evaluate the model on test data
h = sigmoid(np.dot(test_X, W1) + b1)
y_pred = softmax(np.dot(h, W2) + b2)
accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(test_y, axis=1))
test_accuracy_one_hidden_adam.append(accuracy)
print("Test Accuracy (One Hidden Layer with Adam Optimizer) = {:.2f}%".format(accuracy * 100))

        
plt.plot(loss_history_one_hidden_adam , color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve (One Hidden Layer with adam)")
plt.show()

print("\n\n")

plt.plot(loss_history_one_hidden_adam[::469] , color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve (One Hidden Layer with adam)")
plt.show()

"""**Plotting curve for each case**"""

#Plot the learning curves for each case
plt.plot(loss_history_one_hidden[::469], label="One Hidden Layer" , color='green')
plt.plot(loss_history_one_hidden_momentum[::469], label="One Hidden Layer with Momentum" , color='blue')
plt.plot(loss_history_one_hidden_adam[::469], label="One Hidden Layer with Adam Optimizer" , color='red')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.show()

print("\n")

# Plot the test accuracies for each case
plt.bar(["One Hidden Layer", "Momentum", "Adam Optimizer"], 
        [test_accuracy_one_hidden[0], test_accuracy_one_hidden_momentum[0], test_accuracy_one_hidden_adam[0]], )

plt.ylim([0.9, 1.0])
plt.xlabel("Model")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracies")
plt.show()

