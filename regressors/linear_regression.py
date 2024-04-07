from sklearn.linear_model import LinearRegression

from regressors.base import RegressionModelEnvironment
from support.data_utils import get_diabetes_data
from support.models_base import IModel
from support.train_utils import train


class LinearRegressionModelWrapper(IModel):
    def __init__(self, regressor: LinearRegression):
        self.model = regressor

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x) -> any:
        return self.model.predict(x)


def main():
    data_set = get_diabetes_data()

    model = LinearRegressionModelWrapper(LinearRegression())
    model_environment = RegressionModelEnvironment(model, data_set, random_state=1)
    train(model_environment)


if __name__ == '__main__':
    main()
