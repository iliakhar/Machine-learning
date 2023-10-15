from copy import deepcopy

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Utils import *


def start_point():
    filename = "Regression/winequalityN.csv"
    df: pd.DataFrame = get_data_frame(filename)

    win_col: list[int] = [0, 1, 2]
    for i in range(3):
        md_df = deepcopy(df)
        if win_col[i] != 2:
            md_df = md_df[md_df["type"] == win_col[i]]
        x = md_df.drop(columns='quality')
        y = md_df["quality"]
        print(f"Iteration: {i}")
        prepare_to_avg: list = []

        for j in range(10):
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, stratify=y)

            median = SimpleImputer(missing_values=-1, strategy='median')
            scaler = StandardScaler()
            num_pipe = make_pipeline(median, scaler)
            x_train = num_pipe.fit_transform(x_train)
            x_test = num_pipe.fit_transform(x_test)

            reg = ElasticNetCV(cv=10, n_jobs=5)
            reg.fit(x_train, y_train)
            y_test = np.ravel(y_test)
            y_pred = reg.predict(x_test)
            success = 0
            for ind in range(len(y_pred)):
                y_data = y_test[ind]
                yp = y_pred[ind]
                if np.abs(y_data - yp) < 1:
                    success += 1
            prepare_to_avg.append(success / len(y_test) * 100)
            print(prepare_to_avg[j])
        avg = np.average(prepare_to_avg)
        print(f"Average: {avg}")
