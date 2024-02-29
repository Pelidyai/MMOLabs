from sklearn.tree import DecisionTreeClassifier

from classifiers.base import ClassifierModelEnvironment
from support.data_utils import get_iris_data
from support.plot_utils import plot_iris_data_set
from support.train_utils import train


def main(should_plot=False):
    data_set = get_iris_data()
    if should_plot:
        plot_iris_data_set(data_set)

    print("First")
    model = DecisionTreeClassifier(criterion='gini')
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)

    print("\nSecond")
    model = DecisionTreeClassifier(criterion='entropy')
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)

    print("\nThird")
    model = DecisionTreeClassifier(criterion='log_loss')
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)


if __name__ == '__main__':
    main()
