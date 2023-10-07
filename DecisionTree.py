from sklearn.metrics import accuracy_score

from Selection import *
from sklearn import tree
from sklearn.model_selection import cross_val_score
import graphviz

class DecisionTree:
    def __init__(self):
        self.treeInfo: dict = {'Depth': [], 'Leaf': [], 'Train': [], 'Test': []}
        self.selection: Selection = None


    def TrainDecisionTree(self):
        self.treeInfo['Train'].append(0)
        self.treeInfo['Depth'].append(0)
        self.treeInfo['Leaf'].append(0)

        clf: tree.DecisionTreeClassifier
        # clf: RandomForestClassifier
        for maxDepth in range(1, 21):
            for minLeaf in range(1, 10):
                clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=maxDepth,
                                                  min_samples_leaf=minLeaf)

                # clf = RandomForestClassifier(criterion='entropy', max_depth=maxDepth,
                #                                   min_samples_leaf=minLeaf)

                scores = cross_val_score(estimator=clf, cv=7, X=self.selection.x_train,
                                         y=self.selection.y_train, n_jobs=-1)

                meanAccur = scores.mean()
                if meanAccur > self.treeInfo['Train'][-1]:
                    self.treeInfo['Train'][-1] = meanAccur
                    self.treeInfo['Depth'][-1] = maxDepth
                    self.treeInfo['Leaf'][-1] = minLeaf

    def TestDecisionTree(self):
        self.treeInfo['Test'].append(0)
        clf = tree.DecisionTreeClassifier(criterion='entropy',
                                          max_depth=self.treeInfo['Depth'][-1],
                                          min_samples_leaf=self.treeInfo['Leaf'][-1])

        # clf = RandomForestClassifier(criterion='entropy',
        #                                   max_depth=treeInfo['Depth'][-1],
        #                                   min_samples_leaf=treeInfo['Leaf'][-1])
        clf.fit(self.selection.x_train, self.selection.y_train)
        tst = clf.predict(self.selection.x_test)
        self.treeInfo['Test'][-1] = accuracy_score(self.selection.y_test, tst)
        self.SaveTreeToFile(clf, 'treesPdf/tree'+str(len(self.treeInfo['Test'])))


    def SaveTreeToFile(self, clf: tree.DecisionTreeClassifier, fileName: str):
        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=None, filled=True,
                                        rounded=True)  # Important parameters can be customized
        graph = graphviz.Source(dot_data)
        graph.render(view=False, format="pdf", filename=fileName)