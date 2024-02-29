from sklearn.naive_bayes import GaussianNB

from classifiers.base import ClassifierModelEnvironment
from support.data_utils import get_breast_cancer_data
from support.train_utils import train


def main():
    data_set = get_breast_cancer_data()

    model = GaussianNB()
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)


if __name__ == '__main__':
    main()
