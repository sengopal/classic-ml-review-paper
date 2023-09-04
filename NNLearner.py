from collections import defaultdict

import pandas as pd
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import time

from ClassifierType import ClassifierType

rcParams.update({'figure.autolayout': True})


# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
class NNLearner(ClassifierType):
    # TODO: parameters for max_depth and leaf_size and manage accuracy with pruning
    def __init__(self, data, target, data_name):
        super().__init__(data, target, data_name)
        self.clz_name = "neural"

    def gen_validation_curves(self, possible_range_dict):
        self.create_validation_curve(MLPClassifier(max_iter=500), param_name="hidden_layer_sizes", param_value=possible_range_dict['hidden_layer_sizes'])
        self.create_validation_curve(MLPClassifier(max_iter=500), param_name="alpha", param_value=possible_range_dict['alpha'])
        self.create_validation_curve(MLPClassifier(max_iter=500, solver='sgd'), param_name="momentum", param_value=possible_range_dict['momentum'])
        self.create_validation_curve(MLPClassifier(max_iter=500, solver='sgd'), param_name="learning_rate_init", param_value=possible_range_dict['learning_rate_init'])
        self.create_validation_curve(MLPClassifier(max_iter=500), param_name="solver", param_value=possible_range_dict['solver'])

    def init_classifier(self, optimal_values):
        optimal_values['random_state'] = 3
        self.classifier = MLPClassifier(**optimal_values)
        return self.classifier

    def generate_epoch_curve(self, optimal_values):
        optimal_values['max_iter'] = 10000
        optimal_values['warm_start'] = False
        mlp = MLPClassifier(**optimal_values)

        epoch_X_train, epoch_X_test, epoch_y_train, epoch_y_test = train_test_split(self.X_train, self.y_train, test_size=0.2)
        d = defaultdict(dict)

        for epoch in range(0, 6000):
            c, _ = self.get_labels()
            start_time = time.time() * 1000
            mlp.partial_fit(epoch_X_train, epoch_y_train, classes=c)
            train_time = (time.time() * 1000) - start_time
            start_time = time.time() * 1000
            train_error = 1.0 - mlp.score(epoch_X_train, epoch_y_train)
            test_error = 1.0 - mlp.score(epoch_X_test, epoch_y_test)
            test_time = (time.time() * 1000) - start_time
            d['train'][epoch] = train_error
            d['test'][epoch] = test_error
            d['train_t'][epoch] = train_time
            d['test_t'][epoch] = test_time

            if epoch % 100 == 0:
                print("{},{},{}, {}, {}".format(epoch, train_error, test_error, train_time, test_time))
        df = pd.DataFrame(data=d)
        df.to_csv('./output/results/{}/epoch_chart_{}.csv'.format(self.data_name, self.clz_name))
        # df = pd.read_csv('./output/results/rattle/epoch_chart_neural.csv')
        self.plot_epoch_curve(df)
