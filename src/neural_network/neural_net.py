from typing import Literal
import numpy as np
from activation_function import (
    leaky_relu_backward,
    relu,
    leaky_relu,
    relu_backward,
    sigmoid,
    sigmoid_backward,
    softmax,
    tanh,
    tanh_backward,
)
from optimizers import initialize_adam, initialize_velocity, random_mini_batches


class NeuralNetwork:
    def __init__(
        self,
        layers_dims: list,
        activations: list[Literal["relu", "sigmoid", "leaky_relu", "softmax", "tanh"]],
        lr=0.00075,
        epochs=100,
        mini_batch_size=64,
        beta=0.9,
    ):
        """
        layers_dims: array of number of neurons in each layer in the network
        """
        assert len(activations) == len(layers_dims) - 1, (
            f"Number of activation functions ({len(activations)}) "
            f"must match the number of layers minus one ({len(layers_dims) - 1})."
        )
        self.activations = activations
        self.layers_dims = layers_dims
        self.lr = lr
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.beta = beta
        self.parameters = self.__initialize_parameters_deep()
        self.velocity = initialize_velocity(self.parameters)
        self.activations_forward = [
            self.__get_activation_function(act) for act in activations
        ]

    def __linear_forward(self, A, W, b):
        """
        Linear part forward propagation.
        """
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def __linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)
        finding the gradient
        """
        A_prev, W, _ = cache
        m = A_prev.shape[1]

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def cost(self, AL, Y):
        """
        Cross-entropy cost function for softmax output.
        """
        m = Y.shape[1]
        AL = np.clip(AL, 1e-10, 1 - 1e-10)  # Clipping to avoid log(0)
        cost = -np.sum(Y * np.log(AL)) / m
        return np.squeeze(cost)

    def __get_activation_function(self, activation):
        """
        Returns the corresponding activation function for the given string.
        """
        activation_functions = {
            "relu": relu,
            "sigmoid": sigmoid,
            "leaky_relu": leaky_relu,
            "softmax": softmax,
            "tanh": tanh,
        }

        if activation in activation_functions:
            return activation_functions[activation]
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def __initialize_parameters_deep(self):
        """
        Initialize weights and biases for each layer in the network.
        """
        np.random.seed(3)
        parameters = {}
        L = len(self.layers_dims)  # number of layers in the network

        for l in range(1, L):
            parameters["W" + str(l)] = (
                np.random.randn(self.layers_dims[l], self.layers_dims[l - 1]) * 0.01
            )

            parameters["b" + str(l)] = np.zeros((self.layers_dims[l], 1))

        return parameters

    def __forward_activation(self, A_prev, W, b, activation):
        Z, linear_cache = self.__linear_forward(A_prev, W, b)
        A = activation(Z)
        cache = (linear_cache, Z)
        return A, cache

    def _activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        Z = activation_cache

        if activation == relu:
            dZ = relu_backward(dA, Z)
        elif activation == sigmoid:
            dZ = sigmoid_backward(dA, Z)
        elif activation == tanh:
            dZ = tanh_backward(dA, Z)
        elif activation == leaky_relu:
            dZ = leaky_relu_backward(dA, Z)
        elif activation == softmax:
            dZ = dA  # Special case for softmax, derivative is dA = AL - Y

        dZ = np.clip(dZ, -1, 1)

        dA_prev, dW, db = self.__linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def forward_pass(self, X):
        """
        Forward propagation through the network.
        """
        L = len(self.layers_dims)  # number of layers in the network
        caches = []
        A = X
        for l in range(1, L):
            W, b = self.parameters[f"W{l}"], self.parameters[f"b{l}"]
            A_prev = A
            A, cache = self.__forward_activation(
                A_prev, W, b, self.activations_forward[l - 1]
            )
            caches.append(cache)

        AL = A
        return AL, caches

    def backward_pass(self, AL, Y, caches):
        """
        Backward propagation through the network.
        """
        grads = {}
        L = len(caches)  # number of layers in the network
        # m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        # Compute the derivative for softmax
        dAL = AL - Y

        # Backprop through last layer
        current_cache = caches[L - 1]
        dA_prev, dW, db = self._activation_backward(
            dAL, current_cache, self.activations_forward[L - 1]
        )

        grads[f"dW{L}"] = dW
        grads[f"db{L}"] = db
        grads[f"dA{L - 1}"] = dA_prev

        # Backprop through remaining layers
        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev, dW, db = self._activation_backward(
                dA_prev, current_cache, self.activations_forward[l]
            )
            grads[f"dA{l}"] = dA_prev
            grads[f"dW{l + 1}"] = dW
            grads[f"db{l + 1}"] = db

        return grads

    def update_parameters(self, grads):
        """
        Update parameters using gradient descent.
        """
        L = len(self.parameters) // 2  # number of layers in the network

        for l in range(L):
            self.parameters[f"W{l+1}"] -= self.lr * grads[f"dW{l+1}"]
            self.parameters[f"b{l+1}"] -= self.lr * grads[f"db{l+1}"]

        return self.parameters

    def update_parameters_with_momentum(self, grads):
        L = len(self.parameters) // 2
        for l in range(1, L + 1):
            self.velocity["dW" + str(l)] = (
                self.beta * self.velocity["dW" + str(l)]
                + (1 - self.beta) * grads["dW" + str(l)]
            )
            self.velocity["db" + str(l)] = (
                self.beta * self.velocity["db" + str(l)]
                + (1 - self.beta) * grads["db" + str(l)]
            )
            self.parameters["W" + str(l)] -= self.lr * self.velocity["dW" + str(l)]
            self.parameters["b" + str(l)] -= self.lr * self.velocity["db" + str(l)]

    def update_parameters_with_adam(
        self, grads, v, s, t, beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
        L = len(self.parameters) // 2
        v_corrected = {}
        s_corrected = {}

        # Perform Adam update on all parameters
        for l in range(1, L + 1):
            # Moving average of the gradients
            v["dW" + str(l)] = (
                beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
            )
            v["db" + str(l)] = (
                beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
            )

            # Compute bias-corrected first moment estimate
            v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1**t)
            v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1**t)

            # Moving average of the squared gradients
            s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (
                grads["dW" + str(l)] ** 2
            )
            s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (
                grads["db" + str(l)] ** 2
            )

            # Compute bias-corrected second moment estimate
            s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2**t)
            s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2**t)

            # Update parameters
            self.parameters["W" + str(l)] -= (
                self.lr
                * v_corrected["dW" + str(l)]
                / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
            )
            self.parameters["b" + str(l)] -= (
                self.lr
                * v_corrected["db" + str(l)]
                / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)
            )

        return self.parameters, v, s

    def fit(self, X, Y):
        """
        Train the neural network.
        """
        np.random.seed(1)
        costs = []

        for i in range(0, self.epochs):
            AL, caches = self.forward_pass(X)
            cost = self.cost(AL, Y)
            grads = self.backward_pass(AL, Y, caches)
            self.parameters = self.update_parameters(grads)

            # # Print the cost every 200 iterations
            if i % 200 == 0:
                print(f"Cost after iteration {i}: {cost}")
                costs.append(cost)

        return costs

    def fit_momentum(self, X, Y):
        np.random.seed(1)
        costs = []
        for i in range(self.epochs):
            mini_batches = random_mini_batches(X, Y, self.mini_batch_size)
            for mini_batch in mini_batches:
                (mini_batch_X, mini_batch_Y) = mini_batch
                AL, caches = self.forward_pass(mini_batch_X)
                cost = self.cost(AL, mini_batch_Y)
                grads = self.backward_pass(AL, mini_batch_Y, caches)
                self.update_parameters_with_momentum(grads)
            if i % 200 == 0:
                print(f"Cost after epoch {i}: {cost}")
                costs.append(cost)
        return costs

    def fit_adam(self, X, Y):
        """
        Train the neural network using Adam optimization algorithm.
        """
        np.random.seed(1)
        costs = []
        t = 0  # initializing the counter for Adam updates
        v, s = initialize_adam(self.parameters)

        for i in range(self.epochs):
            mini_batches = random_mini_batches(X, Y, self.mini_batch_size)
            for mini_batch in mini_batches:
                (mini_batch_X, mini_batch_Y) = mini_batch
                AL, caches = self.forward_pass(mini_batch_X)
                cost = self.cost(AL, mini_batch_Y)
                grads = self.backward_pass(AL, mini_batch_Y, caches)

                t += 1  # Increment the Adam counter
                self.parameters, v, s = self.update_parameters_with_adam(grads, v, s, t)

            if i % 200 == 0:
                print(f"Cost after epoch {i}: {cost}")
                costs.append(cost)

        return costs

    def predict(self, X):
        AL, _ = self.forward_pass(X)
        predictions = np.argmax(AL, axis=0)
        return predictions

    def accuracy(self, X, Y):
        predictions = self.predict(X)
        labels = np.argmax(Y, axis=0)
        accuracy = np.mean(predictions == labels) * 100
        return accuracy
