from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from tqdm import tqdm


def load_data(path):
    data = sio.loadmat(path)
    return data['X'], data['X_test'], data['y']


def my_log_loss(y, s):
    t1 = y.T @ np.log(s)
    t2 = (~y.astype(bool)).astype(int).T @ np.log(1 - s)
    return -float(t1 + t2)


def plot_costs(costs_df, fig_path):
    fig, ax = plt.subplots()
    costs_df.plot(ax=ax)
    ax.set_xlabel('iteration')
    ax.set_ylabel('cost function (log loss)')
    ax.set_ylim(0, float(costs_df.max().max()) * 1.1)
    ax.grid()
    fig.savefig(fig_path)


class LogisticRegression():

    def __init__(self, C, lr, lr_decay=False, max_iter=1e2, solver='sgd'):
        self.C = C
        self.lr = lr
        self.lr_decay = lr_decay
        self.solver = solver
        self.max_iter = max_iter
        self.solver = solver
        if lr_decay:
            self.lr_init = lr

    def fit(self, x_tr, y_tr, verbose=False):
        # initialize
        ones = np.ones((x_tr.shape[0], 1))
        x_tr = np.hstack((x_tr, ones))
        self.w = np.zeros((x_tr.shape[1], 1))

        costs = defaultdict(list)

        for i in tqdm(range(self.max_iter)):
            # output prob
            s = expit(x_tr @ self.w)

            # cost function
            if verbose:
                cost_train = my_log_loss(y_tr, s)
                costs[f'train_{self.solver}_decay_{self.lr_decay}'].append(cost_train)

            # learning rate decay
            if self.lr_decay:
                self.lr = self.lr_init / (i + 1)

            # update weight
            if self.solver == 'bgd':
                self.w = self.w - self.lr * (2 * self.C * self.w - x_tr.T @ (y_tr - s))
            if self.solver == 'sgd':
                rand_i = np.random.randint(0, x_tr.shape[0], 1)
                self.w = self.w - self.lr * (2 * self.C * self.w - (y_tr[rand_i] - s[rand_i]) * x_tr[rand_i].T)

        if verbose:
            return pd.DataFrame(costs)

    def predict_proba(self, x):
        ones = np.ones((x.shape[0], 1))
        x = np.hstack((x, ones))
        return expit(x @ self.w).ravel()

    def predict(self, x):
        prob = self.predict_proba(x)
        return np.round(prob).ravel()

    def score(self, x, y):
        pred = self.predict(x)
        return accuracy_score(y, pred)
