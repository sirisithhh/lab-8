import numpy as np
import matplotlib.pyplot as plt

# ====== A1.a: Summation Unit ======
def summation_unit(inputs, weights):
    return np.dot(inputs, weights)

# ====== A1.b: Activation Units ======
def step_activation(x):
    return 1 if x >= 0 else 0

def bipolar_step_activation(x):
    return 1 if x >= 0 else -1

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    return np.tanh(x)

def relu_activation(x):
    return max(0, x)

def leaky_relu_activation(x):
    return x if x >= 0 else 0.01 * x

# ====== A1.c: Comparator Unit ======
def comparator_unit(expected, actual):
    return expected - actual

# ====== A2: Perceptron learning function (custom) ======
def perceptron_learning(X, y, weights, learning_rate, activation_function, epochs=1000, error_threshold=0.002):
    errors = []
    n_samples = len(y)
    for epoch in range(epochs):
        total_error = 0
        for inputs, target in zip(X, y):
            summation = summation_unit(inputs, weights)
            output = activation_function(summation)
            error = comparator_unit(target, output)
            weights += learning_rate * error * np.array(inputs)
            total_error += error ** 2
        errors.append(total_error)
        if total_error <= error_threshold:
            break
    return weights, errors

# ====== Main: Experiment with AND gate (A2) ======
X = np.array([
    [1, 0, 0],  # bias, A, B
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])
y = [0, 0, 0, 1]  # AND gate output

initial_weights = np.array([10, 0.2, -0.75])
learning_rate = 0.05

# Train with step activation
final_weights, errors = perceptron_learning(X, y, initial_weights, learning_rate, step_activation)
print("Final weights:", final_weights)
print("Epochs to convergence:", len(errors))

# Plot error vs epochs (A2)
plt.plot(errors)
plt.xlabel('Epochs')
plt.ylabel('Sum Square Error')
plt.title('Perceptron Learning with Step Activation on AND Gate')
plt.show()

# ====== A3: Compare activation functions ======
act_funcs = [step_activation, bipolar_step_activation, sigmoid_activation, relu_activation, leaky_relu_activation]
act_names = ['Step', 'BipolarStep', 'Sigmoid', 'ReLU', 'LeakyReLU']
converge_epochs = []

for act_func in act_funcs:
    _, errors = perceptron_learning(X, y, initial_weights.copy(), learning_rate, act_func)
    converge_epochs.append(len(errors))

plt.bar(act_names, converge_epochs, color='skyblue')
plt.ylabel('Iterations to converge')
plt.title('Iterations to converge for different activation functions')
plt.show()

# ====== A4: Vary learning rate ======
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lr_epochs = []

for lr in learning_rates:
    _, errors = perceptron_learning(X, y, initial_weights.copy(), lr, step_activation)
    lr_epochs.append(len(errors))

plt.plot(learning_rates, lr_epochs, 'o-')
plt.xlabel('Learning Rate')
plt.ylabel('Iterations to converge')
plt.title('Iterations to converge vs Learning Rate (Step Activation)')
plt.show()

# ====== Tabulate Results (A5) ======
print("\n=== A5: Convergence Table ===")
print("Activation     Iterations to Converge")
for act, epoch in zip(act_names, converge_epochs):
    print(f"{act:<12} {epoch}")

print("\nLearning Rate  Iterations to Converge")
for lr, epoch in zip(learning_rates, lr_epochs):
    print(f"{lr:<13} {epoch}")
