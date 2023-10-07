import numpy as np
from DecisionTree import *
import matplotlib.pyplot as plt


def GetDataFrame():
    dataFrame: pd.DataFrame = pd.read_csv('heart_data.csv')

    colTypes = dict.fromkeys(dataFrame.columns, np.int64)

    for col in dataFrame.columns:
        if dataFrame[col].dtype == np.object_:
            dataFrame[col] = dataFrame[col].str.replace('?', '-1')
    dataFrame = dataFrame.astype(colTypes)
    return dataFrame


def main():
    dataFrame: pd.DataFrame = GetDataFrame()
    # treeInfo: dict = {'Depth': [], 'Leaf': [], 'Train': [], 'Test':[]}
    decisTree: DecisionTree = DecisionTree()
    for i in range(10):
        decisTree.selection = Selection(dataFrame)
        # selection:Selection = GetSelection(dataFrame)
        decisTree.TrainDecisionTree()
        decisTree.TestDecisionTree()

    resultFrame: pd.DataFrame = pd.DataFrame.from_dict(decisTree.treeInfo)
    resultFrame.to_csv('Result.csv')
    print(resultFrame)

    # resultFrame = pd.read_csv('Result.csv')
    # print(resultFrame)



if __name__ == "__main__":
    main()