# from DecisionTree import *
from ElasticNet import *
from Utils import *


def main():
    start_point()
    # filename = 'heart_data.csv'
    # dataFrame: pd.DataFrame = GetDataFrame(filename)
    # # treeInfo: dict = {'Depth': [], 'Leaf': [], 'Train': [], 'Test':[]}
    # decisTree: DecisionTree = DecisionTree()
    # for i in range(10):
    #     decisTree.selection = Selection(dataFrame)
    #     # selection:Selection = GetSelection(dataFrame)
    #     decisTree.TrainDecisionTree()
    #     decisTree.TestDecisionTree()
    #
    # resultFrame: pd.DataFrame = pd.DataFrame.from_dict(decisTree.treeInfo)
    # resultFrame.to_csv('Result.csv')
    # print(resultFrame)
    #
    # resultFrame = pd.read_csv('Result.csv')
    # print(resultFrame)


if __name__ == "__main__":
    main()
