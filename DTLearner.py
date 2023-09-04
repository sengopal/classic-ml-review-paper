import pydotplus
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn.tree import export_graphviz

from ClassifierType import ClassifierType


# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
class DTLearner(ClassifierType):

    def __init__(self, data, target, data_name):
        super().__init__(data, target, data_name)
        self.clz_name = "dtlearn"

    def draw_plot(self, image_loc):
        try:
            # https://stackoverflow.com/questions/28160129/is-there-a-way-to-retrieve-the-final-number-of-nodes-generated-by-sklearn-tree-d
            ifs_count = len([x for x in self.classifier.tree_.feature if x != _tree.TREE_UNDEFINED])
            n_nodes = self.classifier.tree_.node_count
            print("decision_count: {}, node_count: {}".format(ifs_count, n_nodes))

            dot_data = export_graphviz(self.classifier)
            graph = pydotplus.graph_from_dot_data(dot_data)
            graph.write_png(image_loc)
            Image(graph.create_png())
        except:
            print("GraphViz's executables not found")

    def gen_validation_curves(self, possible_range_dict):
        self.create_validation_curve(DecisionTreeClassifier(), param_name="max_depth", param_value=possible_range_dict['max_depth'])
        self.create_validation_curve(DecisionTreeClassifier(), param_name="min_samples_split", param_value=possible_range_dict['min_samples_split'])

    def init_classifier(self, optimal_values):
        optimal_values['random_state'] = 3
        self.classifier = DecisionTreeClassifier(**optimal_values)
        return self.classifier

    def get_regular_classifier(data_name):
        if data_name == 'rattle':
            noprune_values = {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 250}
        elif data_name == 'wildfire':
            noprune_values = {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 0.01}
        return DecisionTreeClassifier(**noprune_values)

    def get_pruned_classifier(data_name):
        if data_name == 'rattle':
            noprune_values = {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 250}
            pruned_values = noprune_values.copy()
            pruned_values['max_depth'] = pruned_values['max_depth'] - 2
            pruned_values['min_samples_leaf'] = 2
        elif data_name == 'wildfire':
            noprune_values = {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 0.01}
            pruned_values = noprune_values.copy()
            pruned_values['max_depth'] = pruned_values['max_depth'] - 1
            pruned_values['min_samples_leaf'] = 1
        return DecisionTreeClassifier(**pruned_values)