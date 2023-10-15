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


