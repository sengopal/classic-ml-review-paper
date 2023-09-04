import sqlite3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import seaborn as sns
import sklearn
from matplotlib import rcParams
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from BoostLearner import BoostLearner
from DTLearner import DTLearner
from DummyLearner import DummyLearner
from KNNLearner import KNNLearner
from NNLearner import NNLearner
from SVMLearner import SVMLearner

rcParams.update({'figure.autolayout': True})

GRID_SEARCH = False
VALIDATION_CURVE = False
EPOCH_GRAPH = False

DATA_FOLDER = "./data/"

def read_toy_data(data_name):
    # http://archive.ics.uci.edu/ml/datasets/Iris
    if data_name == 'iris':
        columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
        df = pd.read_csv(DATA_FOLDER + "iris/iris.data", header=None, names=columns)
        df = preprocess_file(df)
        X = df.iloc[:, 0:-1]
        y = df['class']
        return X, y, columns
    #     TODO: Class distribution graph
    elif data_name == 'car':
        columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        df = pd.read_csv(DATA_FOLDER + "car/car.data", header=None, names=columns)
        df = preprocess_file(df)
        #  https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
        onehotencoder = OneHotEncoder()
        X = df.iloc[:, 0:-1]
        X = onehotencoder.fit_transform(X).toarray()
        y = df['class']
        return X, y, columns
    #     TODO: Class distribution graph
    elif data_name == 'mm':
        columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        df = pd.read_csv(DATA_FOLDER + "car/car.data", header=None, names=columns)
        df = preprocess_file(df)
        #  https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
        onehotencoder = OneHotEncoder()
        X = df.iloc[:, 0:-1]
        X = onehotencoder.fit_transform(X).toarray()
        y = df['class']
        print(X.shape)
        return X, y, columns


def read_rattle_data():
    sns.set(style="whitegrid", palette="RdBu_r")
    IMG_PATH = './images/rattle/pre-process/'
    # some pre-processing steps from https://www.kaggle.com/rodrigofrb/feature-selections-and-classification-models
    df = pd.read_csv(DATA_FOLDER + "rattle/weatherAUS.csv", header=0)
    df = preprocess_file(df)
    # As instructed in the data set for RISK_MM.
    # Dropping the date column as there is no co-relation
    # Evaporation,Sunshine,Cloud9am,Cloud3pm - has nearly 60k entries with NaN
    # df = df.drop(['RISK_MM', 'Date', 'Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis=1)
    df = df.drop(['Location', 'RISK_MM', 'Date'], axis=1)

    df['RainToday'].replace('No', 0, inplace=True)
    df['RainToday'].replace('Yes', 1, inplace=True)
    df['RainTomorrow'].replace('No', 0, inplace=True)
    df['RainTomorrow'].replace('Yes', 1, inplace=True)

    plt.figure()
    # Count distribution by Object variable
    sns.countplot(y='RainTomorrow', data=df, order=df['RainTomorrow'].value_counts(normalize=True).index)
    plt.savefig(IMG_PATH + 'RainTomorrow.png')

    plt.figure()
    # Count distribution by Object variable
    sns.countplot(y='WindGustDir', data=df, order=df['WindGustDir'].value_counts(normalize=True).index)
    plt.savefig(IMG_PATH + 'WindGustDir.png')

    plt.figure()
    # sns.lineplot(x=df.index.values, y="Rainfall", data=df, order=df['Rainfall'].value_counts(normalize=True).index)
    sns.distplot(df['Evaporation'])
    plt.savefig(IMG_PATH + 'Evaporation.png')

    plt.figure()
    sns.distplot(df['Sunshine'])
    plt.savefig(IMG_PATH + 'Sunshine.png')

    # dummy variables for WinGustDir and similar labelled attributes
    # else we get ValueError: could not convert string to float: 'SSW'
    df = pd.get_dummies(df)

    X = df.drop('RainTomorrow', axis=1)
    y = df['RainTomorrow']

    scaler = MinMaxScaler(feature_range=[0, 1])  # trying instead of StandardScaler
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    model = SelectKBest(score_func=chi2, k=30)  # Using k = 30 to get top 30 features - Totally 65 features are available
    model.fit(X, y)
    np.set_printoptions(precision=3)

    # Process feature scores into a sorted dataframe with the top 30 attributes.
    us_scores_df = pd.DataFrame(model.scores_).transpose()
    us_scores_df.columns = list(X)
    us_scores_df = us_scores_df.transpose()
    us_scores_df.sort_values(0, ascending=False, inplace=True)
    # print(us_scores_df.head(30))  # returns the top 30 attributes by score.

    # selecting only the top 30 features
    X = X[list(us_scores_df.head(30).transpose())]
    print("Data shape: ", X.shape)
    return X, y, list(X)


# Reads data file, preprocess and returns data, target
def read_wildfire_data():
    IMG_PATH = './images/wildfire/pre-process/'
    conn = sqlite3.connect("./data/wildfire/FPA_FOD_20170508.sqlite")
    # Forming a smaller DF set
    # SOURCE_SYSTEM_TYPE, SOURCE_SYSTEM, **REPORTING** columns do not influence wildfire and should not be considered
    # FIRE_CODE --> Cost information --> not necessary
    # In order to observe a multi-class dataset, the most common classes are used and the rest of the instances are dropped
    df = pd.read_sql_query(
        "SELECT FIRE_SIZE as fire_size, FIRE_SIZE_CLASS as fire_size_class, DISCOVERY_DATE as discovery_date, DISCOVERY_TIME as discovery_time, CONT_DATE as cont_date, CONT_TIME as cont_time, STAT_CAUSE_CODE as stat_cause_code, FIPS_NAME as county_name from Fires where STATE = 'CA' AND CONT_TIME IS NOT NULL AND DISCOVERY_TIME is not null AND FIPS_CODE IS NOT NULL",
        conn)

    # 5- Debris - 5
    # 9 - Misc - 9 + 13 (missing)
    # 14 - Human Error -  "Campfire"(4) + "Children"(8) + "Smoking" (3) + "Fireworks" (10)
    # 2 - Equipment Failure - "Equipment Use" (2) + "Railroad" (6) + "Powerline" (11) + "Structure" (12)
    # 1 - Lightning - 1
    # 7 - Arson - 7

    df.loc[df['stat_cause_code'].isin([4, 8, 3, 10]), 'stat_cause_code'] = 14.0
    df.loc[df['stat_cause_code'].isin([2, 6, 11, 12]), 'stat_cause_code'] = 2.0
    # Using only four classes of data
    df = df[df['stat_cause_code'].isin([1, 2, 7, 14])]
    print(df.shape)

    df = df.sample(frac=0.5)

    sns.set(style="darkgrid", palette="Set3")

    plt.figure()
    # Count distribution by Object variable
    sns.countplot(y='stat_cause_code', data=df, order=df['stat_cause_code'].value_counts(normalize=True).index)
    plt.savefig(IMG_PATH + 'stat_cause_code.png')

    df['fire_size'] = df.fire_size.astype(float)

    # Extracting month and weekdays
    df['disc_clean_date'] = pd.to_datetime(df['discovery_date'] - pd.Timestamp(0).to_julian_date(), unit='D')
    df['discovery_month'] = df['disc_clean_date'].dt.strftime('%b')
    df['discovery_weekday'] = df['disc_clean_date'].dt.strftime('%a')
    df['disc_date_final'] = pd.to_datetime(df.disc_clean_date.astype('str') + ' ' + df.discovery_time.astype('str'), errors='coerce')

    # Number of hours to containment
    df['cont_clean_date'] = pd.to_datetime(df['cont_date'] - pd.Timestamp(0).to_julian_date(), unit='D')
    df['cont_date_final'] = pd.to_datetime(df.cont_clean_date.astype('str') + ' ' + df.cont_time, errors='coerce')
    df['time_to_cont'] = (df.cont_date_final - df.disc_date_final).astype('timedelta64[m]')
    # Fill NaN with mean values instead of deleting rows
    df['time_to_cont'] = df.time_to_cont.fillna(df.time_to_cont.mean())

    plt.figure()
    plt.tight_layout()
    fig1 = sns.regplot(x='fire_size', y='time_to_cont', data=df, scatter=True).figure.savefig(IMG_PATH + 'fire_size.png')

    # Create column for time_of_day_grp categories
    time_of_day_grp = []
    df['discovery_time'] = pd.to_numeric(df.discovery_time, errors='coerce')
    for row in df.discovery_time:
        if row >= 0 and row < 400:
            time_of_day_grp.append('early_morning')
        elif row >= 400 and row < 800:
            time_of_day_grp.append('mid_morning')
        elif row >= 800 and row < 1200:
            time_of_day_grp.append('late_morning')
        elif row >= 1200 and row < 1600:
            time_of_day_grp.append('afternoon')
        elif row >= 1600 and row < 2000:
            time_of_day_grp.append('evening')
        elif row >= 2000 and row < 2400:
            time_of_day_grp.append('night')
        else:
            time_of_day_grp.append('n_a')
    df['time_of_day_grp'] = time_of_day_grp

    plt.figure(figsize=(8, 4))
    sns.countplot(x='time_of_day_grp', data=df, order=df['time_of_day_grp'].value_counts(normalize=True).index, )
    plt.savefig(IMG_PATH + 'time_of_day_grp.png')

    plt.figure()
    # Distribution by Weekday
    sns.countplot(x='discovery_weekday', data=df, order=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
    plt.savefig(IMG_PATH + 'discovery_weekday.png')

    plt.figure()
    # Distribution by Month
    sns.countplot(x='discovery_month', data=df, order=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.savefig(IMG_PATH + 'discovery_month.png')

    # DROP UNEEDED COLUMNS
    df.drop(['discovery_date', 'discovery_time', 'disc_clean_date', 'disc_date_final', 'cont_date', 'cont_time', 'cont_clean_date', 'cont_date_final'], axis=1, inplace=True)

    X = df.drop('stat_cause_code', axis=1)
    y = df['stat_cause_code']

    # Label Encoding
    labelencoder = LabelEncoder()
    encode_columns = ['fire_size_class', 'county_name', 'discovery_month', 'discovery_weekday', 'time_of_day_grp']
    for col in encode_columns:
        X[col] = labelencoder.fit_transform(X[col].astype(str))

    scale_columns = ['fire_size', 'time_to_cont']
    scaler = MinMaxScaler(feature_range=[0, 1])  # trying instead of StandardScaler
    X[scale_columns] = scaler.fit_transform(X[scale_columns])

    print("Data shape: ", X.shape)
    return X, y, list(X)


def read_data(data_name):
    if data_name == 'rattle':
        return read_rattle_data()
    elif data_name == 'wildfire':
        return read_wildfire_data()

def preprocess_file(df):
    # print("NA count: {na_count}".format(na_count=df.isna().sum(0)))
    pre_instances = df.count().sum(0)
    df = df.dropna()
    # print("Instances dropped: {count}".format(count=(pre_instances - df.count().sum(0))))
    return df

def dt_learner_no_pruning(data_name):
    # Step 1: Pre-process data to form datasets
    print("DT_LEARNER_NO_PRUNE: ", data_name)
    X, y, columns = read_data(data_name)
    dt = DTLearner(X, y, data_name)

    # Step 2: Complexity Analysis
    if data_name == 'rattle':
        possible_range_dict = {'max_depth': np.arange(1, 25, 1), 'min_samples_split': np.linspace(0.001, 0.18, 10)}  # np.arange(2, 300, 10)
        if VALIDATION_CURVE:
            dt.gen_validation_curves(possible_range_dict)

    elif data_name == 'wildfire':
        possible_range_dict = {'max_depth': np.arange(1, 25, 1), 'min_samples_split': np.linspace(0.001, 0.3, 10)}
        if VALIDATION_CURVE:
            dt.gen_validation_curves(possible_range_dict)

    # Step 3: Grid Search to find the best optimal values
    if data_name == 'rattle':
        # For 'rattle' the hyper parameter ranges are identified as below
        # max_features - inconsequential and min_samples_split:[250,300] , max_depth: 4-7
        param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(4, 8), 'min_samples_split': [0.1, 0.12]}
        if GRID_SEARCH:
            optimal_values = dt.grid_search(DecisionTreeClassifier(), param_grid)
        else:
            optimal_values = {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 0.1}
    elif data_name == 'wildfire':
        # For 'wildfire' the hyper parameter ranges are identified as below
        # max_features - inconsequential and min_samples_split:[0.01-0.10] , max_depth: 15-20
        param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(3, 7), 'min_samples_split': np.linspace(0.01, 0.2, 10)}
        if GRID_SEARCH:
            optimal_values = dt.grid_search(DecisionTreeClassifier(), param_grid)
        else:
            optimal_values = {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 0.01}

    print('optimal_values:', optimal_values)
    # Use identified parameters for classifier training
    print(dt.init_classifier(optimal_values))
    # Step 4:  Learning Curve Analysis on model with optimal hyperparameter selection from step
    dt.generate_learning_curve()

    # # Step 5: Run Fit with optimized model parameters on Training Data and Predict on Test data
    dt.train()
    dt.draw_plot("./images/{}/dtlearn_noprune_tree.png".format(data_name))
    dt.predict()

def plot_multi_line_learning_curve(train_sizes, clz1_train_scores, clz1_test_scores, clz2_train_scores, clz2_test_scores, data_name, clz_name):
    plt.figure(figsize=(10, 5))
    # Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    plt.title("Cross Validation")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    labels = []
    if clz_name == 'dtlearn_prune_comp':
        labels = ["Training score - no pruning", "Test score without Pruning", "Training score With Pruning", "Test score With Pruning"]
    elif clz_name == 'boost_dtlearn_comp':
        labels = ["DT - Training score", "DT - Test score", "Boost - Training score", "Boost - Test score"]

    plt.plot(train_sizes, clz1_train_scores, 'o-', color="r", label=labels[0], alpha=0.5, linestyle='dashed')
    plt.plot(train_sizes, clz1_test_scores, 'o-', color="g", label=labels[1], alpha=0.5, linestyle='dashed')

    plt.plot(train_sizes, clz2_train_scores, 'o-', color="purple", label=labels[2], alpha=0.5)
    plt.plot(train_sizes, clz2_test_scores, 'o-', color="darkblue", label=labels[3], alpha=0.5)
    plt.legend(loc="best")
    plt.grid()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig("./images/{}/{}_learning_curve.png".format(data_name, clz_name))


def dt_learner_with_pruning(data_name):
    # Step 1: Pre-process data to form datasets
    print("DT_LEARNER_WITH_PRUNE: ", data_name)
    X, y, columns = read_data(data_name)
    dt_no_prune = DTLearner(X, y, data_name)
    dt_pruned = DTLearner(X, y, data_name)

    # No Pruning Parameters
    if data_name == 'rattle':
        noprune_values = {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 0.1}
    elif data_name == 'wildfire':
        noprune_values = {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 0.01}

    # https://en.wikipedia.org/wiki/Decision_tree_pruning
    # Pruning is a technique in machine learning and search algorithms that reduces the size of decision trees by removing sections of the tree that provide little power to classify instances.
    # Pruning reduces the complexity of the final classifier, and hence improves predictive accuracy by the reduction of overfitting.

    ## Using pruning construct a new DT classifier and compare the performance measures of both using confusion matrix and accuracy

    # Pre-Pruning Techniques
    # Tree depth would be pre pruning
    # Leaf size affects overfitting
    # min_samples_leaf - The minimum number of samples required to be at a leaf node and current default is 1
    # A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches.

    if data_name == 'rattle':
        pruned_values = noprune_values.copy()
        pruned_values['max_depth'] = pruned_values['max_depth'] - 2
        pruned_values['min_samples_leaf'] = 2
    elif data_name == 'wildfire':
        pruned_values = noprune_values.copy()
        pruned_values['max_depth'] = pruned_values['max_depth'] - 1
        pruned_values['min_samples_leaf'] = 1

    print("No Prune: ", dt_no_prune.init_classifier(noprune_values))
    dt_no_prune.train()
    dt_no_prune.draw_plot("./images/{}/dtlearn_noprune_tree.png".format(data_name))
    dt_no_prune.predict(conf_mat_name="conf_mat_no_prune")

    print("Pruned: ", dt_pruned.init_classifier(pruned_values))
    dt_pruned.train()
    dt_pruned.draw_plot("./images/{}/dtlearn_with_prune_tree.png".format(data_name))
    dt_pruned.predict(conf_mat_name="conf_mat_with_prune")

    # Learning Curve Analysis on model with optimal hyperparameter selection from step
    df_no_prune = dt_no_prune.generate_learning_curve()
    df_pruned = dt_pruned.generate_learning_curve()
    plot_multi_line_learning_curve(df_no_prune['train_size'], df_no_prune['train_score'], df_no_prune['test_score'],  df_pruned['train_score'], df_pruned['test_score'], data_name, 'dtlearn_prune_comp')


def nn_learner(data_name):
    print("NN_LEARNER: ", data_name)
    # Step 1: Pre-process data to form datasets
    X, y, columns = read_data(data_name)
    nn = NNLearner(X, y, data_name)

    # Step 2: Complexity Analysis
    if data_name == 'rattle':
        # same number of neurons as there are features in our data set
        possible_range_dict = {
            'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,), (30, 30), (30, 30, 30), (30, 30, 30, 30, 30), ],
            'alpha': np.around(np.linspace(0.0001, 0.001, 10), 4),
            'momentum': np.linspace(0.1, 1.0, 10),
            'learning_rate_init': np.linspace(0.001, 0.1, 10),  # Testing for learning_rate : {‘constant’}
            'solver': ['lbfgs', 'sgd', 'adam']
        }
        if VALIDATION_CURVE:
            nn.gen_validation_curves(possible_range_dict)

    elif data_name == 'wildfire':
        possible_range_dict = {
            'hidden_layer_sizes': [(7,), (100,), (100, 100, 100), (7, 7, 7), (7, 21, 7), (21,), (49, 49, 49), (7, 7, 7, 7, 7), (14, 7, 7, 7, 7, 14)],
            'alpha': np.around(np.linspace(0.0001, 0.001, 10), 4),
            'momentum': np.linspace(0.1, 1.0, 10),
            'learning_rate_init': np.linspace(0.001, 0.1, 10),  # Testing for learning_rate : {‘constant’}
            'solver': ['lbfgs', 'sgd', 'adam']

        }
        if VALIDATION_CURVE:
            nn.gen_validation_curves(possible_range_dict)

    # Step 3: Grid Search to find the best optimal values
    if data_name == 'rattle':
        # For 'rattle' the hyper parameter ranges are identified as below
        # hidden_layer_sizes - (50, 100, 50)
        # alpha: around 0.012
        param_grid = {'hidden_layer_sizes': [(30, 30)], 'alpha': [0.0001, 0.0003], 'activation': ['logistic', 'tanh', 'relu']}
        if GRID_SEARCH:
            optimal_values = nn.grid_search(MLPClassifier(max_iter=500), param_grid)
        else:
            optimal_values = {'activation': 'logistic', 'alpha': 0.0003, 'hidden_layer_sizes': (30, 30), 'max_iter': 1000, 'solver': 'adam'}
    elif data_name == 'wildfire':
        # For 'wildfire' the hyper parameter ranges are identified as below
        param_grid = {'hidden_layer_sizes': [(49, 49, 49), (100, 100, 100)], 'alpha': [0.0001, 0.0007], 'activation': ['logistic', 'tanh', 'relu']}
        if GRID_SEARCH:
            optimal_values = nn.grid_search(MLPClassifier(max_iter=500), param_grid)
        else:
            optimal_values = {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (100, 100, 100), 'max_iter': 1000, 'solver': 'adam'}

    print('optimal_values:', optimal_values)
    # Use identified parameters for classifier training
    print(nn.init_classifier(optimal_values))
    #
    # # Step 4:  Learning Curve Analysis on model with optimal hyperparameter selection from step
    nn.generate_learning_curve()

    if EPOCH_GRAPH:
        nn.generate_epoch_curve(optimal_values)
    # # # Step 5: Run Fit with optimized model parameters on Training Data and Predict on Test data
    nn.train()
    nn.predict()


def svm_learner(data_name):
    print("SVM_LEARNER: ", data_name)
    # Step 1: Pre-process data to form datasets
    X, y, columns = read_data(data_name)
    svm = SVMLearner(X, y, data_name)

    # Step 2: Complexity Analysis
    if data_name == 'rattle':
        possible_range_dict = {
            'C': [0.001, 0.10, 0.1, 10, 25, 50, 75, 100, 200, 500, 750, 1000, 1250, 1500],
            'gamma': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 5, 10, 100]
        }
        if VALIDATION_CURVE:
            svm.gen_validation_curves(possible_range_dict)

    elif data_name == 'wildfire':
        possible_range_dict = {
            'C': [0.001, 0.10, 0.1, 10, 25, 50, 75, 100, 200, 500, 750, 1000, 1250, 1500],
            'gamma': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 5, 10, 100]
        }
        if VALIDATION_CURVE:
            svm.gen_validation_curves(possible_range_dict)

    # Step 3: Grid Search to find the best optimal values
    if data_name == 'rattle':
        # For 'rattle' the hyper parameter ranges are identified as below
        param_grid = {'C': [10, 25, 50], 'gamma': [0.01, 0.1, 0.2]}
        if GRID_SEARCH:
            optimal_values = svm.grid_search(SVC(), param_grid)
        else:
            optimal_values = {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
    elif data_name == 'wildfire':
        # For 'wildfire' the hyper parameter ranges are identified as below
        param_grid = {'C': [10], 'kernel': ['rbf'], 'gamma': [0.01, 0.1, 0.2]}
        if GRID_SEARCH:
            optimal_values = svm.grid_search(SVC(), param_grid)
        else:
            optimal_values = {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}

    print('optimal_values:', optimal_values)

    for kernel in ['linear', 'rbf', 'sigmoid']:
        print('kernel:' + kernel)
        optimal_values['kernel'] = kernel
        optimal_values['max_iter'] = 1000000
        # Use identified parameters for classifier training
        print(svm.init_classifier(optimal_values))
        # Step 4:  Learning Curve Analysis on model with optimal hyperparameter selection from step
        svm.generate_learning_curve(kernel)
        # # Step 5: Run Fit with optimized model parameters on Training Data and Predict on Test data

    # Resetting for Train and fit
    optimal_values['kernel'] = 'rbf'
    svm.init_classifier(optimal_values)
    svm.train()
    svm.predict()

    if EPOCH_GRAPH:
        svm.generate_epoch_curve(optimal_values)

def knn_learner(data_name):
    print("KNN_LEARNER: ", data_name)
    # Step 1: Pre-process data to form datasets
    X, y, columns = read_data(data_name)
    knn = KNNLearner(X, y, data_name)

    # Step 2: Complexity Analysis
    if data_name == 'rattle':
        # Generate validation analysis for rbf and different gamma values since default kernel='rbf'
        possible_range_dict = {
            'n_neighbors': np.arange(10, 105, 5),
            'p': np.arange(1, 6),
            'leaf_size': np.arange(1, 30, 3),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        if VALIDATION_CURVE:
            knn.gen_validation_curves(possible_range_dict)

    elif data_name == 'wildfire':
        possible_range_dict = {
            'n_neighbors': np.arange(5, 500, 10),
            'p': np.arange(1, 10),
            'leaf_size': np.arange(1, 30, 3),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        if VALIDATION_CURVE:
            knn.gen_validation_curves(possible_range_dict)

    # Step 3: Grid Search to find the best optimal values
    if data_name == 'rattle':
        # For 'rattle' the hyper parameter ranges are identified as below
        # weights: uniform, p = 1, leaf_size and algorithm are inconsequential
        param_grid = {'n_neighbors': np.arange(70, 140, 20)}
        if GRID_SEARCH:
            optimal_values = knn.grid_search(KNeighborsClassifier(p=1, weights='uniform'), param_grid)
        else:
            optimal_values = {'n_neighbors': 90, 'p': 1, 'weights': 'uniform'}
    elif data_name == 'wildfire':
        # For 'wildfire' the hyper parameter ranges are identified as below
        # n_neighbors = [500 - 1000]
        # 'algorithm' and 'leaf_size' is inconsequential, so GRID_SEARCH here
        optimal_values = {'n_neighbors': 245, 'p': 1, 'weights': 'uniform'}

    print('optimal_values:', optimal_values)
    # Use identified parameters for classifier training
    print(knn.init_classifier(optimal_values))

    # Step 4:  Learning Curve Analysis on model with optimal hyperparameter selection from step
    knn.generate_learning_curve()

    # # Step 5: Run Fit with optimized model parameters on Training Data and Predict on Test data
    knn.train()
    knn.predict()


def boost_learner(data_name, pruned=False):
    if pruned:
        print("BOOST_LEARNER_PRUNED: ", data_name)
    else:
        print("BOOST_LEARNER: ", data_name)
    # Step 1: Pre-process data to form datasets
    X, y, columns = read_data(data_name)
    clz_name = "boost_pruned" if pruned else "boost_regular"
    boost = BoostLearner(X, y, data_name, clz_name)

    # Step 2: Complexity Analysis
    if data_name == 'rattle':
        # Generate validation analysis for rbf and different gamma values since default kernel='rbf'
        possible_range_dict = {
            'n_estimators': list(range(10, 500, 20)),
            'learning_rate': np.linspace(0.05, 0.5, 30),
        }
        if VALIDATION_CURVE:
            boost.gen_validation_curves(possible_range_dict)

    elif data_name == 'wildfire':
        possible_range_dict = {
            'n_estimators': list(range(100, 10000, 1000)),
            'learning_rate': np.linspace(0.1, 10.0, 50),
        }
        if VALIDATION_CURVE:
            boost.gen_validation_curves(possible_range_dict)

    # Step 3: Grid Search to find the best optimal values
    base_estimator = DTLearner.get_pruned_classifier(data_name) if pruned else DTLearner.get_regular_classifier(data_name)
    if data_name == 'rattle':
        param_grid = {'n_estimators': list(range(10, 100, 10)), 'learning_rate': np.linspace(0.05, 0.3, 6)}
        if GRID_SEARCH:
            optimal_values = boost.grid_search(AdaBoostClassifier(base_estimator=base_estimator), param_grid)
        else:
            optimal_values = {'learning_rate': 0.15, 'n_estimators': 90}
    elif data_name == 'wildfire':
        # For 'wildfire' the hyper parameter ranges are identified as below
        param_grid = {'n_estimators': list(range(50, 250, 50)), 'learning_rate': [0.05, 0.1, 0.2]}
        if GRID_SEARCH:
            optimal_values = boost.grid_search(AdaBoostClassifier(base_estimator=base_estimator), param_grid)
        else:
            optimal_values = {'learning_rate': 0.05, 'n_estimators': 200} if pruned else {'learning_rate': 0.1, 'n_estimators': 50}

    print('optimal_values:', optimal_values)
    optimal_values['base_estimator'] = base_estimator
    # Use identified parameters for classifier training
    print(boost.init_classifier(optimal_values))

    # Step 4:  Learning Curve Analysis on model with optimal hyperparameter selection from step
    boost.generate_learning_curve()

    # # Step 5: Run Fit with optimized model parameters on Training Data and Predict on Test data
    boost.train()
    boost.predict()


def run_classifier(clz, values):
    clz.init_classifier(values)
    clz.train()
    clz.predict()


def get_default_accuracy(data_name):
    print("Default Accuracy: ", data_name)
    X, y, columns = read_data(data_name)

    run_classifier(DTLearner(X, y, data_name), {})
    run_classifier(NNLearner(X, y, data_name), {})
    run_classifier(KNNLearner(X, y, data_name), {})
    run_classifier(SVMLearner(X, y, data_name), {})
    run_classifier(BoostLearner(X, y, data_name, 'boost_regular'), values={'base_estimator': DecisionTreeClassifier()})


def get_dummy_classifier_results(data_name):
    print("Dummy Accuracy: ", data_name)
    X, y, columns = read_data(data_name)
    run_classifier(DummyLearner(X, y, data_name), {})


def generate_boost_learner_prune_comp(data_name):
    dt_file_name = "learn_curve_{data_name}_dtlearn_".format(data_name=data_name)
    boost_file_name = "learn_curve_{data_name}_boost_pruned_".format(data_name=data_name)
    dt_file_data = np.load('./output/results/{}/{}.npz'.format(data_name, dt_file_name))
    boost_data = np.load('./output/results/{}/{}.npz'.format(data_name, boost_file_name))

    plot_multi_line_learning_curve(dt_file_data['train_sizes'], dt_file_data['train_scores'], dt_file_data['test_scores'], boost_data['train_scores'], boost_data['test_scores'], data_name, 'boost_dtlearn_comp')


if __name__ == "__main__":
    print("pd.__version__", pd.__version__)
    print("np.__version__", np.__version__)
    print("sklearn.__version__", sklearn.__version__)
    print("seaborn.__version__", seaborn.__version__)
    print("matplotlib.__version__", matplotlib.__version__)

    np.random.seed(100)

    # Get Accuracy for default Classifiers
    get_default_accuracy('rattle')
    get_default_accuracy('wildfire')

    # Get Accuracy for Dummy Classifiers
    get_dummy_classifier_results('rattle')
    get_dummy_classifier_results('wildfire')

    # # Decision Tree Analysis
    dt_learner_no_pruning('rattle')
    dt_learner_no_pruning('wildfire')

    # Decision Tree Analysis - with pruning
    dt_learner_with_pruning('rattle')
    dt_learner_with_pruning('wildfire')

    # Boosting
    boost_learner('rattle')
    boost_learner('rattle',pruned=True)
    generate_boost_learner_prune_comp('rattle')

    boost_learner('wildfire')
    boost_learner('wildfire', pruned=True)
    generate_boost_learner_prune_comp('wildfire')

    # KNN Analysis
    knn_learner('rattle')
    knn_learner('wildfire')

    # Neural Network Analysis
    nn_learner('rattle')
    nn_learner('wildfire')

    # SVM Analysis
    svm_learner('rattle')
    svm_learner('wildfire')

#
# def test_iris():
#     n_classes = 3
#     plot_colors = "ryb"
#     plot_step = 0.02
#
#     # Load data
#     # https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html#sphx-glr-auto-examples-tree-plot-iris-dtc-py
#     iris = load_iris()
#
#     for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
#                                     [1, 2], [1, 3], [2, 3]]):
#         # We only take the two corresponding features
#         X = iris.data[:, pair]
#         y = iris.target
#
#         # Train
#         clf = DecisionTreeClassifier().fit(X, y)
#
#         # Plot the decision boundary
#         plt.subplot(2, 3, pairidx + 1)
#
#         x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#         y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#         xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#                              np.arange(y_min, y_max, plot_step))
#         plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
#
#         Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#         Z = Z.reshape(xx.shape)
#         cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
#
#         plt.xlabel(iris.feature_names[pair[0]])
#         plt.ylabel(iris.feature_names[pair[1]])
#
#         # Plot the training points
#         for i, color in zip(range(n_classes), plot_colors):
#             idx = np.where(y == i)
#             plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
#                         cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
#
#     plt.suptitle("Decision surface of a decision tree using paired features")
#     plt.legend(loc='lower right', borderpad=0, handletextpad=0)
#     plt.axis("tight")
#
#     plt.figure()
#     clf = DecisionTreeClassifier().fit(iris.data, iris.target)
#     plot_tree(clf, filled=True)
#     plt.show()
