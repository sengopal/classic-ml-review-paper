from collections import defaultdict
from sklearn import svm
from ClassifierType import ClassifierType
import pandas as pd
from sklearn.model_selection import train_test_split
import time


# https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
class SVMLearner(ClassifierType):
    # TODO: parameters for max_depth and leaf_size and manage accuracy with pruning
    def __init__(self, data, target, data_name):
        super().__init__(data, target, data_name)
        self.clz_name = "svm"

    def gen_validation_curves(self, possible_range_dict, cv_folds=3):
        self.create_validation_curve(svm.SVC(), param_name="C", param_value=possible_range_dict['C'], cv_folds=cv_folds)
        self.create_validation_curve(svm.SVC(), param_name="gamma", param_value=possible_range_dict['gamma'], cv_folds=cv_folds)

    def init_classifier(self, optimal_values):
        optimal_values['random_state'] = 3
        self.classifier = svm.SVC(**optimal_values)
        return self.classifier

    def generate_epoch_curve(self, optimal_values):
        epoch_X_train, epoch_X_test, epoch_y_train, epoch_y_test = train_test_split(self.X_train, self.y_train, test_size=0.2)
        d = defaultdict(dict)

        for epoch in range(100, 8000, 100):
            c, _ = self.get_labels()
            optimal_values['max_iter'] = epoch
            svc = svm.SVC(**optimal_values)

            start_time = time.time() * 1000
            svc.fit(epoch_X_train, epoch_y_train)
            train_time = (time.time() * 1000) - start_time
            start_time = time.time() * 1000

            train_error = 1.0 - svc.score(epoch_X_train, epoch_y_train)
            test_error = 1.0 - svc.score(epoch_X_test, epoch_y_test)
            test_time = (time.time() * 1000) - start_time
            d['train'][epoch] = train_error
            d['test'][epoch] = test_error
            d['train_t'][epoch] = train_time
            d['test_t'][epoch] = test_time
            if epoch % 100 == 0:
                print("{},{},{}, {}, {}".format(epoch, train_error, test_error, train_time, test_time))

        df = pd.DataFrame(data=d)
        df.to_csv('./output/results/{}/epoch_chart_{}.csv'.format(self.data_name, self.clz_name))
        # df = pd.read_csv('./output/results/rattle/epoch_chart_svm.csv')
        self.plot_epoch_curve(df)
