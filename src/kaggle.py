from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from utils import load_data, plot_costs, LogisticRegression
from save_csv import results_to_csv


def cross_validation(x_train, y_train, params, n_splits, random_state):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = defaultdict(list)

    for tr_idx, va_idx in skf.split(x_train, y_train):
        x_tr, x_va = x_train[tr_idx], x_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model = LogisticRegression(**params)
        model.fit(x_tr, y_tr)

        scores['acc_tr'].append(model.score(x_tr, y_tr))
        scores['acc_va'].append(model.score(x_va, y_va))

    return pd.DataFrame(scores)


if __name__ == "__main__":

    # set seed
    seed = 289
    np.random.seed = seed

    # load data
    data_path = Path('data/data.mat')
    x_train, x_test, y_train = load_data(data_path)

    # parameters
    params = {
        'C': 1,
        'lr': 1e-5,
        'lr_decay': False,
        'max_iter': int(1e6),
        'solver': 'bgd'
    }

    # preprocessing
    scaler = StandardScaler()
    scaler.fit(np.vstack((x_train, x_test)))
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # evaluate
    scores_df = cross_validation(x_train, y_train, params=params, n_splits=5, random_state=seed)
    print(scores_df)
    print(scores_df.mean(axis=0))

    # predict
    model = LogisticRegression(**params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # save
    results_to_csv(y_pred)
