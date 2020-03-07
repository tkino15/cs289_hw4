import numpy as np
import pandas as pd
from pathlib import Path

from utils import load_data, LogisticRegression, plot_costs


if __name__ == "__main__":
    # set seed
    np.random.seed = 289

    # laod data
    data_path = Path('data/data.mat')
    x_train, x_test, y_train = load_data(data_path)

    # parameters
    bgd_params = {
        'C': 1,
        'lr': 1e-7,
        'max_iter': int(1e5),
        'solver': 'bgd'
    }
    sgd_params = {
        'C': 1,
        'lr': 1e-5,
        'max_iter': int(1e5),
        'solver': 'sgd'
    }
    decay_params = {
        'C': 1,
        'lr': 1e-3,
        'lr_decay': True,
        'max_iter': int(1e5),
        'solver': 'sgd'
    }

    # models
    model = LogisticRegression(**bgd_params)
    bgd_costs_df = model.fit(x_train, y_train, verbose=True)

    model = LogisticRegression(**sgd_params)
    sgd_costs_df = model.fit(x_train, y_train, verbose=True)

    model = LogisticRegression(**decay_params)
    decay_costs_df = model.fit(x_train, y_train, verbose=True)

    # plot
    costs_df = pd.concat([bgd_costs_df, sgd_costs_df, decay_costs_df], axis=0)
    fig_path = Path('figures/3_2.png')
    plot_costs(costs_df, fig_path)
