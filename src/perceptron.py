import numpy as np

class Perceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        n_features = X.shape[1]
        rgen = np.random.RandomState(25)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=n_features)
        self.b = 0.0  # зміщення
        self.errors_ = []  # кількість помилок на кожній епосі (для графіка)

        for epoch in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                y_hat = self.predict_one(xi)
                update = self.eta * (target - y_hat)  # 0, якщо відповідь правильна
                self.w += update * xi
                self.b += update
                if update != 0:
                    errors += 1
            self.errors_.append(errors)
        return self

    def predict_one(self, x):
        return 1 if (np.dot(x, self.w) + self.b) > 0 else 0

    def predict(self, X):
        # X @ self.w рахує w·x для всіх рядків одразу (матрично)
        return np.where(X @ self.w + self.b > 0, 1, 0)