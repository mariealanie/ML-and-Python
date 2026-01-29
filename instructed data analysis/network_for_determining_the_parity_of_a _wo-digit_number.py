import numpy as np

X = np.array([[int(b) for b in f"{i:06b}"] for i in range(64)])
y = np.array([[1 if i % 2 == 0 else 0] for i in range(64)])   

np.random.seed(42)
weights_input_hidden = 2 * np.random.random((6, 2)) - 1 
weights_hidden_output = 2 * np.random.random((2, 1)) - 1 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

for epoch in range(10000):
    hidden = sigmoid(np.dot(X, weights_input_hidden))
    output = sigmoid(np.dot(hidden, weights_hidden_output))

    error = y - output

    d_output = error * sigmoid_derivative(output)
    d_hidden = d_output.dot(weights_hidden_output.T) * sigmoid_derivative(hidden)

    weights_hidden_output += hidden.T.dot(d_output) * 0.1
    weights_input_hidden += X.T.dot(d_hidden) * 0.1

    if epoch % 2000 == 0:
        acc = np.mean((output > 0.5) == y)
        print(f"Эпоха {epoch}: точность = {acc*100:.2f}%")

preds = (output > 0.5).astype(int)
accuracy = np.mean(preds == y)
print("\nТочность на всех 64 числах:", accuracy * 100, "%")
