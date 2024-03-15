from sklearn.tree import DecisionTreeClassifier

from classifiers.base import ClassifierModelEnvironment
from support.data_utils import get_iris_data
from support.models_base import IModel
from support.plot_utils import plot_iris_data_set
from support.train_utils import train


class ClassifierTreeModelWrapper(IModel):
    def __init__(self, tree: DecisionTreeClassifier):
        self.model = tree

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x) -> any:
        return self.model.predict(x)


def main(should_plot=False):
    data_set = get_iris_data()
    if should_plot:
        plot_iris_data_set(data_set)

    print("Metric - gini, to 1 object in leaf")
    model = ClassifierTreeModelWrapper(DecisionTreeClassifier(criterion='gini'))
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)

    print("\nMetric - gini, to 10 object in leaf")
    model = ClassifierTreeModelWrapper(DecisionTreeClassifier(criterion='gini', min_samples_leaf=10))
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)

    print("\nMetric - entropy")
    model = ClassifierTreeModelWrapper(DecisionTreeClassifier(criterion='entropy'))
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)

    print("\nMetric - log_loss")
    model = ClassifierTreeModelWrapper(DecisionTreeClassifier(criterion='log_loss'))
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)


if __name__ == '__main__':
    main()
