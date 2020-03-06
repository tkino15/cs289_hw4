import numpy as np


def logistic_function(w, X):
    return 1 / (1 + np.e ** -(X @ w))


def update_weight(w, X, y, s):
    Omega = np.diag(s * (1 - s).T)
    return w + np.linalg.solve((X.T @ Omega @ X), X.T @ (y - s))


def newton_logreg(w, X, y, iter, verbose=True):
    for i in range(iter):
        # output
        s = logistic_function(w, X)

        if verbose:
            print(f'[{i}] w: {w}')
            print(f'[{i}] s: {s}')

        # update weight
        w = update_weight(w, X, y, s)

    return s, w


if __name__ == "__main__":
    # test case
    X = np.array([[0.2, 3.1], [1, 3], [-0.2, 1.2], [1, 1.1]])
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((X, ones))
    w = np.array([-1, 1, 0])
    y = np.array([1, 1, 0, 0])

    # logistic regression with newton method
    s, w = newton_logreg(w, X, y, iter=5, verbose=True)
