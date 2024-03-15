from sklearn.neighbors import KNeighborsClassifier

from classifiers.base import ClassifierModelEnvironment
from support.data_utils import get_iris_data
from support.models_base import IModel
from support.plot_utils import plot_iris_data_set
from support.train_utils import train


class KNeighborsModelWrapper(IModel):
    def __init__(self, classifier: KNeighborsClassifier):
        self.model = classifier

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x) -> any:
        return self.model.predict(x)


def main(should_plot=False):
    data_set = get_iris_data()
    if should_plot:
        plot_iris_data_set(data_set)

    print("K = 2")
    model = KNeighborsModelWrapper(KNeighborsClassifier(n_neighbors=2))
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)

    print("\nK = 5")
    model = KNeighborsModelWrapper(KNeighborsClassifier(n_neighbors=5))
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)

    print("\nK = 10")
    model = KNeighborsModelWrapper(KNeighborsClassifier(n_neighbors=10))
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)


if __name__ == '__main__':
    main()
