from sklearn.linear_model import LogisticRegression

from classifiers.base import ClassifierModelEnvironment
from support.data_utils import get_breast_cancer_data
from support.train_utils import train


def main():
    data_set = get_breast_cancer_data()

    # max=10_000 needed for optimizer correct work
    print("Solver - liblinear, penalty - l1")
    model = LogisticRegression(solver='liblinear', penalty='l1', max_iter=10_000)
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)

    print("\nSolver - liblinear, penalty - l2")
    model = LogisticRegression(solver='liblinear', max_iter=10_000)
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)

    print("\nSolver - newton-cg")
    model = LogisticRegression(solver='newton-cg', max_iter=10_000)
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)

    print("\nSolver - lbfgs")
    model = LogisticRegression(max_iter=10_000)
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)


if __name__ == '__main__':
    main()
