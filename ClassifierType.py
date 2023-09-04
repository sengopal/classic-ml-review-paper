import time
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve

rcParams.update({'figure.autolayout': True})

class ClassifierType:
    @abstractmethod
    def __init__(self, data, target, data_name):
        self.data = data
        self.target = target
        self.data_name = data_name
        # X_test and y_test are the final hold out set for validating the model created
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, target, test_size=0.2)

        # to be set during run time
        self.predictions = None
        self.classifier = None
        self.clz_name = None

    @abstractmethod
    def init_classifier(self, optimal_values):
        print('init_classifier is called')

    def train(self):
        self.classifier.fit(self.X_train, self.y_train)

    def predict(self, conf_mat_name=None):
        self.predictions = self.classifier.predict(self.X_test)
        classifier_score = self.classifier.score(self.X_test, self.y_test)

        try:
            false_positive_rate, true_positive_rate, thresholds = roc_curve(self.y_test, self.predictions)
            roc_auc = auc(false_positive_rate, true_positive_rate)
        except:
            roc_auc = None

        acc_score = metrics.accuracy_score(self.y_test, self.predictions)
        # Model Precision: what percentage of positive tuples are labeled as such?
        precision = metrics.precision_score(self.y_test, self.predictions, average='weighted')
        # Model Recall: what percentage of positive tuples are labelled as such?
        recall = metrics.recall_score(self.y_test, self.predictions, average='weighted')

        if roc_auc is None:
            print("{}:, score: {:.2f}, accuracy: {:.2f}, roc_auc: None, precision: {:.2f}, recall: {:.2f} ".format(self.clz_name, classifier_score, acc_score, precision, recall))
        else:
            print("{}:, score: {:.2f}, accuracy: {:.2f}, roc_auc: {:.2f}, precision: {:.2f}, recall: {:.2f} ".format(self.clz_name, classifier_score, acc_score, roc_auc, precision, recall))
        # create training and testing vars
        # TODO: Tuning Parameters - https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
        if conf_mat_name is not None:
            self.gen_confusion_matrix(title="Confusion Matrix - {}".format(self.data_name), file_name=conf_mat_name)

    # https://stackoverflow.com/questions/31161637/grid-search-cross-validation-in-sklearn
    def grid_search(self, clf, param_grid, nfolds=3):
        # Splitting the self.X_train once again for grid search cross validation scoring.
        gscv = GridSearchCV(clf, param_grid, cv=nfolds, n_jobs=10, verbose=1)
        grid_X_train, grid_X_test, grid_y_train, grid_y_test = train_test_split(self.X_train, self.y_train, test_size=0.1)
        gscv.fit(grid_X_train, grid_y_train)

        grid_results = pd.DataFrame(gscv.cv_results_)
        grid_results.to_csv('./output/{}_{}_grid_cv_results.csv'.format(self.clz_name, self.data_name), index=False)
        test_score = gscv.score(grid_X_test, grid_y_test)

        with open('./output/grid_search_results.csv', 'a') as f:
            f.write('{},{},{:.2f},{}\n'.format(self.clz_name, self.data_name, test_score, gscv.best_params_))

        return gscv.best_params_

    def create_validation_curve(self, clf, param_name, param_value, cv_folds=5):
        train_scores, valid_scores = validation_curve(clf, self.X_train, self.y_train, param_name, param_value, cv=cv_folds, verbose=1, n_jobs=10)
        file_name = "valid_curve_{data_name}_{clz_name}_{param_name}".format(data_name=self.data_name, clz_name=self.clz_name, param_name=param_name)
        np.savez('./output/results/{}/{}.npz'.format(self.data_name, file_name), train_scores=train_scores, valid_scores=valid_scores)

        plot_valid_curve(param_name, param_value, train_scores, valid_scores, self.data_name, file_name)

    def generate_learning_curve(self, kernel=''):
        cv_size = 3
        file_name = "learn_curve_{data_name}_{clz_name}_{kernel}".format(data_name=self.data_name, clz_name=self.clz_name, kernel=kernel)
        train_sizes = np.arange(0.1, 1.0, 0.1)
        _, train_scores, test_scores = learning_curve(self.classifier, self.X_train, self.y_train, train_sizes=train_sizes, cv=cv_size, shuffle=True, n_jobs=10, verbose=1)
        train_times, test_times = self.get_time_data(train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        d = {
            'train_size': train_sizes,
            'train_score': train_scores_mean,
            'test_score': test_scores_mean,
            'train_times': train_times,
            'test_times': test_times
        }

        np.savez('./output/results/{}/{}.npz'.format(self.data_name, file_name), train_sizes=train_sizes, train_scores=train_scores_mean, test_scores=test_scores_mean, train_time=train_times, test_time=test_times)
        df = pd.DataFrame(data=d)
        df.to_csv('./output/results/{}_df.csv'.format(file_name), index='train_size')
        df = pd.read_csv('./output/results/{}_df.csv'.format(file_name))
        plot_learn_curve(df['train_size'], df['train_score'], df['test_score'], df['train_times'], df['test_times'], self.data_name, file_name, kernel)
        return df

    def get_labels(self):
        if self.data_name == 'rattle':
            return [0, 1], ['No', 'Yes']
        elif self.data_name == 'wildfire':
            return [1, 2, 7, 14], ['Lightning', 'Equipment', 'Arson', ' Human']

    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    def gen_confusion_matrix(self, normalize=False, title=None, cmap=plt.cm.Blues, file_name=None):
        # Compute confusion matrix
        classes, labels = self.get_labels()
        cm = confusion_matrix(self.y_test, self.predictions, classes)
        # Only use the labels that appear in the data
        # classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure()
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=labels, yticklabels=labels,
               title=title,
               ylabel='True label',
               xlabel='Prediction')

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.grid(None)
        if file_name is None:
            plt.show()
        else:
            plt.savefig("./images/{}/{}.png".format(self.data_name, file_name))
        return ax

    np.set_printoptions(precision=2)

    def get_time_data(self, train_sizes):
        train_times = []
        test_times = []
        for size in train_sizes:
            X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, train_size=size)
            start_time = time.time() * 1000
            self.classifier.fit(X_train, y_train)
            train_times.append((time.time() * 1000) - start_time)
            start_time = time.time() * 1000
            self.classifier.predict(X_test)
            test_times.append((time.time() * 1000) - start_time)

        return train_times, test_times

    def plot_epoch_curve(self, df):
        plt.figure(figsize=(9, 5))
        fig, ax1 = plt.subplots(figsize=(10, 5))
        plt.title("{} - Complexity and Time analysis".format(self.data_name), fontsize=16)
        color = 'tab:red'
        ax1.set_xlabel("Training Iterations - Epochs", fontsize=14)
        ax1.set_ylabel("Prediction Error", fontsize=14, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        df = df[df.index % 100 == 0]
        lns1 = ax1.plot(df.index.values, df.train, 'o-', color="r", label=" Training Error", alpha=0.6)
        lns2 = ax1.plot(df.index.values, df.test, 'o-', color="g", label=" Generalization Error", alpha=0.6)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Time (in ms)', color=color)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)

        lns3 = ax2.plot(df.index.values, df.train_t, '--', color="purple", label='Training time', alpha=0.5)
        lns4 = ax2.plot(df.index.values, df.test_t, '--', color="darkblue", label='Testing time', alpha=0.5)
        plt.grid(None)

        lns = lns1 + lns2 + lns3 + lns4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="best")

        plt.savefig("./images/{}/epoch_chart_{}.png".format(self.data_name, self.clz_name))


def plot_learn_curve(train_sizes, train_scores, test_scores, train_times, test_times, data_name, file_name, kernel=None):
    plt.figure(figsize=(10, 5))
    fig, ax1 = plt.subplots(figsize=(10, 5))
    # Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    if kernel is None:
        plt.title("{} - Complexity and Time analysis".format(data_name), fontsize=16)
    else:
        plt.title("{} - Complexity and Time analysis using {} kernel".format(data_name, kernel), fontsize=16)

    ax1.set_xlabel("Training examples", fontsize=14)
    ax1.set_ylabel("Score", fontsize=14, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    lns1 = ax1.plot(train_sizes, train_scores, 'o-', color="r", label="Training score", alpha=0.6)
    lns2 = ax1.plot(train_sizes, test_scores, 'o-', color="g", label="Cross-validation score", alpha=0.6)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Time (in ms)', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)

    lns3 = ax2.plot(train_sizes, train_times, '--', color="purple", label='Training time', alpha=0.5)
    lns4 = ax2.plot(train_sizes, test_times, '--', color="darkblue", label='Testing time', alpha=0.5)
    plt.grid(None)

    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="best")
    plt.savefig("./images/{}/{}.png".format(data_name, file_name))
    print("Learning curve completed ---")


def plot_valid_curve(param_name, param_value, train_scores, valid_scores, data_name, file_name, pruned_train_scores=None, pruned_valid_scores=None):
    plt.figure(figsize=(8, 5))
    plt.tight_layout()
    if len(str(param_value[0])) > 5:
        plt.xticks(rotation=30)

    if isinstance(param_value[0], float) and param_value[0] > 0.01:
        param_value = ['{:.2f}'.format(val) for val in param_value]
    elif isinstance(param_value[0], float) and param_value[0] >= 0.001:
        param_value = ['{:.3f}'.format(val) for val in param_value]

    fig, ax = plt.subplots()

    if pruned_train_scores is None:
        plt.plot([str(val) for val in param_value], np.mean(train_scores, axis=1), 'o-', color="r", label="Training score", alpha=0.55)
        plt.plot([str(val) for val in param_value], np.mean(valid_scores, axis=1), 'o-', color="g", label="Cross-validation score", alpha=0.55)
        plt.legend(loc="best", fontsize=14)
    else:
        plt.plot([str(val) for val in param_value], np.mean(train_scores, axis=1), 'o-', color="r", label="Training score", alpha=0.55, linestyle='dashed')
        plt.plot([str(val) for val in param_value], np.mean(valid_scores, axis=1), 'o-', color="g", label="Cross-validation score", alpha=0.55, linestyle='dashed')
        plt.plot([str(val) for val in param_value], np.mean(pruned_train_scores, axis=1), 'o-', color="purple", label="Pruned - Training score", alpha=0.7)
        plt.plot([str(val) for val in param_value], np.mean(pruned_valid_scores, axis=1), 'o-', color="darkblue", label="Pruned - Cross-validation score", alpha=0.7)
        plt.legend(loc="best", fontsize=10)

    if isinstance(param_value[0], tuple):
        for n, tick in enumerate(ax.xaxis.get_major_ticks()):
            tick.label.set_rotation('vertical')

    if len(param_value) > 10:
        every_nth = 4 if len(param_value) > 15 else 2
        for n, tick in enumerate(ax.xaxis.get_major_ticks()):
            tick.label.set_fontsize(8)
            tick.label.set_visible(False)
            if n % every_nth == 0:
                tick.label.set_visible(True)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().set_title("{} - Validation Curve - {param_name}".format(data_name, param_name=param_name), fontsize=16)
    plt.gca().set_xlabel("{param_name} values".format(param_name=param_name), fontsize=14)
    plt.gca().set_ylabel("Accuracy", fontsize=14)
    plt.savefig("./images/{}/{}.png".format(data_name, file_name))

# if __name__ == "__main__":
# data_name = 'rattle'
# clz_name = 'neural'
# param_name = 'learning_rate_init'
# param_value = np.linspace(0.001, 0.01, 5)
#
# # 'learning_rate_init': np.linspace(0.001, 0.01, 5)
#
# file_name = "valid_curve_{data_name}_{clz_name}_{param_name}".format(data_name=data_name, clz_name=clz_name, param_name=param_name)
# data = np.load('./output/results/{}/{}.npz'.format(data_name, file_name))
# plot_valid_curve(param_name, param_value, data['train_scores'], data['valid_scores'], data_name, file_name)

# data_name = 'rattle'
# clz_name = 'knn'
# file_name = "learn_curve_{data_name}_{clz_name}".format(data_name=data_name, clz_name=clz_name)
# data = np.load('./output/results/{}/{}.npz'.format(data_name, file_name))
# plot_learn_curve(data['train_sizes'], data['train_scores'], data['test_scores'], data_name, file_name)
#
#     def scatter_plot(self):
#         plt.scatter(self.y_test, self.predictions)
#         plt.xlabel("True Values")
#         plt.ylabel("Predictions")
#         plt.show()
