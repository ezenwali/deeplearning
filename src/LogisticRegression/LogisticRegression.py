import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.001, max_iter=100, print_cost=False):
        self.lr = lr
        self.max_iter = max_iter
        self.b = None
        self.w = None
        self.print_cost = print_cost
        self.costs = []

    def __sigmoid(self, z):
        """Calculate the sigmoid for a given value of z

        Args:
            z (Salar or numpy array)

        Returns:
            sigmoid(z)
        """
        return 1 / (1 + np.exp(-z))

    def __propagate(self, X, Y):
        """
        Cost function and its gradient for the propagation

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        grads -- dictionary containing the gradients of the weights and bias
                (dw -- gradient of the loss with respect to w, thus same shape as w)
                (db -- gradient of the loss with respect to b, thus same shape as b)
        cost -- negative log-likelihood cost for logistic regression
        """

        # Forward propagation
        Z = np.dot(self.w.T, X) + self.b  # 𝑤𝑇𝑋+𝑏 linear model

        m = X.shape[1]

        A = self.__sigmoid(Z)  # sigmoid---prediction
        cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))

        # Backward propagation
        dw = (1 / m) * np.dot(X, (A - Y).T)
        db = (1 / m) * np.sum(A - Y)

        grads = {"dw": dw, "db": db}

        cost = np.squeeze(np.array(cost))

        return grads, cost

    def fit(self, X, Y):
        """_summary_ Train the logistic regression model

        Args:
            X -- data of size (num_px * num_px * 3, number of examples)
            Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
        """
        self.w = np.zeros((X.shape[0], 1))
        self.b = 0.0

        for i in range(self.max_iter):
            grads, cost = self.__propagate(X, Y)

            self.w = self.w - self.lr * grads["dw"]
            self.b = self.b - self.lr * grads["db"]

            if i % 100 == 0:
                self.costs.append(cost)

                if self.print_cost:
                    print("Cost after iteration %i: %f" % (i, cost))

    def predict(self, X):
        """_summary_

        Args:
            X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
            Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        """
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))

        A = self.__sigmoid(
            np.dot(self.w.T, X) + self.b
        )  # sigmoid---prediction probability

        assert Y_prediction.shape == A.shape

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        for i in range(A.shape[1]):
            Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

        return Y_prediction

    def predict_prob(self, X):
        return self.__sigmoid(np.dot(self.w.T, X) + self.b)

    def score(self, X, Y):
        Y_prediction = self.predict(X)
        return 1 - np.mean(np.abs(Y_prediction - Y))

    def accuracy(self, Y_test, Y_prediction):
        return 1 - np.mean(np.abs(Y_prediction - Y_test))
