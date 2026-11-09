import numpy as np
from src.perceptron import Perceptron

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    print("Навчені ваги:", ppn.w)
    print("Зміщення (bias):", ppn.b)
    print("Передбачення:", ppn.predict(X))
    print("Помилки по епохах:", ppn.errors_)