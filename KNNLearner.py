from sklearn.neighbors import KNeighborsClassifier

from ClassifierType import ClassifierType


# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
class KNNLearner(ClassifierType):
    def __init__(self, data, target, data_name):
        super().__init__(data, target, data_name)
        self.clz_name = "knn"

    def gen_validation_curves(self, possible_range_dict, cv_folds=3):
        self.create_validation_curve(KNeighborsClassifier(), param_name="n_neighbors", param_value=possible_range_dict['n_neighbors'], cv_folds=cv_folds)
        self.create_validation_curve(KNeighborsClassifier(), param_name="p", param_value=possible_range_dict['p'], cv_folds=cv_folds)
        self.create_validation_curve(KNeighborsClassifier(), param_name="leaf_size", param_value=possible_range_dict['leaf_size'], cv_folds=cv_folds)
        self.create_validation_curve(KNeighborsClassifier(), param_name="weights", param_value=possible_range_dict['weights'], cv_folds=cv_folds)
        self.create_validation_curve(KNeighborsClassifier(), param_name="algorithm", param_value=possible_range_dict['algorithm'], cv_folds=cv_folds)

    def init_classifier(self, optimal_values):
        self.classifier = KNeighborsClassifier(**optimal_values)
        return self.classifier
