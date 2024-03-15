from sklearn.linear_model import LogisticRegression

from classifiers.base import ClassifierModelEnvironment
from support.data_utils import get_breast_cancer_data
from support.models_base import IModel
from support.train_utils import train


class LogisticRegressionModelWrapper(IModel):
    def __init__(self, log_regressor: LogisticRegression):
        self.model = log_regressor

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x) -> any:
        return self.model.predict(x)


def main():
    data_set = get_breast_cancer_data()

    # max=10_000 needed for optimizer correct work
    print("Solver - liblinear, penalty - l1")
    model = LogisticRegressionModelWrapper(LogisticRegression(solver='liblinear', penalty='l1', max_iter=10_000))
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)

    print("\nSolver - liblinear, penalty - l2")
    model = LogisticRegressionModelWrapper(LogisticRegression(solver='liblinear', max_iter=10_000))
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)

    print("\nSolver - newton-cg")
    model = LogisticRegressionModelWrapper(LogisticRegression(solver='newton-cg', max_iter=10_000))
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)

    print("\nSolver - lbfgs")
    model = LogisticRegressionModelWrapper(LogisticRegression(max_iter=10_000))
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)


if __name__ == '__main__':
    main()
