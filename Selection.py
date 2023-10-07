from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd

class Selection:

    def __init__(self):
        pass

    def __init__(self,  dataFrame: pd.DataFrame):
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.SetSelection(dataFrame)

    def SetSelection(self, dataFrame: pd.DataFrame):
        dataLstX = [row[:-1] for row in dataFrame.values.tolist()]
        dataLstY = [row[-1] for row in dataFrame.values.tolist()]

        imp = SimpleImputer(missing_values=-1, strategy="median")
        imp.fit(dataLstX)
        dataLstX = imp.transform(dataLstX)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(dataLstX, dataLstY, train_size=0.7)