from sklearn.naive_bayes import GaussianNB

from classifiers.base import ClassifierModelEnvironment
from support.data_utils import get_breast_cancer_data
from support.models_base import IModel
from support.train_utils import train


class GaussianModelWrapper(IModel):
    def __init__(self, gaussian_model: GaussianNB):
        self.model = gaussian_model

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x) -> any:
        return self.model.predict(x)


def main():
    data_set = get_breast_cancer_data()

    model = GaussianModelWrapper(GaussianNB())
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)


if __name__ == '__main__':
    main()
