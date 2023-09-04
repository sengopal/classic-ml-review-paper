from sklearn.ensemble import AdaBoostClassifier

from DTLearner import DTLearner
from ClassifierType import ClassifierType
from ClassifierType import plot_valid_curve
from sklearn.model_selection import validation_curve
import numpy as np

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
# boosted version of your decision trees - so need the same parameters of the DTLearner
class BoostLearner(ClassifierType):
    def __init__(self, data, target, data_name, clz_name):
        super().__init__(data, target, data_name)
        self.clz_name = clz_name

    def init_classifier(self, optimal_values):
        optimal_values['random_state'] = 3
        self.classifier = AdaBoostClassifier(**optimal_values)
        return self.classifier

    def gen_validation_curves(self, possible_range_dict, cv_folds=3):
        self.gen_validation_curve(AdaBoostClassifier(base_estimator=DTLearner.get_regular_classifier(self.data_name)), AdaBoostClassifier(base_estimator=DTLearner.get_pruned_classifier(self.data_name)), param_name="n_estimators", param_value=possible_range_dict['n_estimators'])
        self.gen_validation_curve(AdaBoostClassifier(base_estimator=DTLearner.get_regular_classifier(self.data_name)), AdaBoostClassifier(base_estimator=DTLearner.get_pruned_classifier(self.data_name)), param_name="learning_rate",param_value=possible_range_dict['learning_rate'])

    def gen_validation_curve(self, clf, clf_prune, param_name, param_value, cv_folds=5):
        file_name = "valid_curve_{data_name}_{clz_name}_{param_name}_noprune".format(data_name=self.data_name, clz_name=self.clz_name, param_name=param_name)
        train_scores, valid_scores = validation_curve(clf, self.X_train, self.y_train, param_name, param_value, cv=cv_folds, verbose=1, n_jobs=20)
        np.savez('./output/results/{}/{}.npz'.format(self.data_name, file_name), train_scores=train_scores, valid_scores=valid_scores)
        print('file_name: '+ file_name)
        data = np.load('./output/results/{}/{}.npz'.format(self.data_name, file_name))
        train_scores, valid_scores = data['train_scores'], data['valid_scores']

        file_name = "valid_curve_{data_name}_{clz_name}_{param_name}_withprune".format(data_name=self.data_name, clz_name=self.clz_name, param_name=param_name)
        pruned_train_scores, pruned_valid_scores = validation_curve(clf_prune, self.X_train, self.y_train, param_name, param_value, cv=cv_folds, verbose=1, n_jobs=20)
        np.savez('./output/results/{}/{}.npz'.format(self.data_name, file_name), train_scores=pruned_train_scores, valid_scores=pruned_valid_scores)
        print('file_name: ' + file_name)
        data = np.load('./output/results/{}/{}.npz'.format(self.data_name, file_name))
        pruned_train_scores, pruned_valid_scores = data['train_scores'], data['valid_scores']

        plot_valid_curve(param_name, param_value, train_scores, valid_scores, self.data_name, file_name, pruned_train_scores, pruned_valid_scores)