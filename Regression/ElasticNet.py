from copy import deepcopy
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Utils import *


class StandartScaler:
    pass


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

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, stratify=y)

        median = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        num_pipe = make_pipeline(median, scaler)
        x_train = num_pipe.fit_transform(x_train)
        x_test = num_pipe.fit_transform(x_test)

        reg = ElasticNetCV(cv=10, random_state=42, n_jobs=5)
        reg.fit(x_train, np.ravel(y_train))
        y_test = np.ravel(y_test)
        y_pred = reg.predict(x_test)
        success = 0
        for ind in range(len(y_pred)):
            y = y_test[ind]
            yp = y_pred[ind]
            if y - yp < 1:
                success += 1
        print(success / len(x_test) * 100)
