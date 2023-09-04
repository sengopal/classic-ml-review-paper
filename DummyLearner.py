from sklearn.dummy import DummyClassifier

from ClassifierType import ClassifierType


# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
class DummyLearner(ClassifierType):

    # TODO: parameters for max_depth and leaf_size and manage accuracy with pruning
    def __init__(self, data, target, data_name):
        super().__init__(data, target, data_name)
        self.clz_name = "dummy"

    def init_classifier(self, optimal_values):
        self.classifier = DummyClassifier(random_state= 3)
        return self.classifier