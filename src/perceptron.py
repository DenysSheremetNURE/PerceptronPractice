import numpy as np

class MyPerceptron:
    def __init__(self, eta=0.01, epochs=10):
        self.eta = eta
        self.epochs = epochs

    def fit(self, X, y):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.errors_ = []

        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                y_pred = self.predict_one(xi)
                update = self.eta * (target - y_pred)
                self.w += update * xi
                self.b += update
                if update != 0:
                    errors += 1
            self.errors_.append(errors)
        return self

    def predict_one(self, x):
        return 1 if (np.dot(x, self.w) + self.b) > 0 else 0

    def predict(self, X):
        return np.where(X @ self.w + self.b > 0, 1, 0)