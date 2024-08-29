import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid of z
    """
    return 1 / (1 + np.exp(-z))


def relu(z):
    """
    Compute the relu of z
    """
    return np.maximum(0, z)


def tanh(z):
    """
    Compute the tanh of z
    """
    return np.tanh(z)


def leaky_relu(z, alpha=0.01):
    """
    Compute the leaky relu of z
    """
    return np.where(z > 0, z, z * alpha)


def softmax(Z):
    """
    Compute the softmax of z
    """
    shift_z = Z - np.max(Z, axis=0, keepdims=True)
    exp_z = np.exp(shift_z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def relu_backward(dA, Z):
    """
    Implement the backward propagation for a ReLU unit.
    """
    dZ = np.array(dA, copy=True)  # Just converting dA to a correct object.

    # When Z <= 0, you should set dZ to 0 as well.
    dZ[Z <= 0] = 0

    assert dZ.shape == Z.shape

    return dZ


def sigmoid_backward(dA, Z):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert dZ.shape == Z.shape

    return dZ


def tanh_backward(dA, Z):
    """
    Implement the backward propagation for a single Tanh unit.
    dZ=dA(1-A**2), where A = Tanh(Z)
    """
    A = np.tanh(Z)
    dZ = dA * (1 - A**2)

    return dZ


def leaky_relu_backward(dA, Z, alpha=0.01):

    dZ = np.array(dA, copy=True)

    # Gradient is 1 where Z > 0, and alpha where Z <= 0
    dZ[Z <= 0] = alpha * dZ[Z <= 0]

    return dZ
