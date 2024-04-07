from sklearn.ensemble import RandomForestRegressor

from regressors.base import RegressionModelEnvironment
from support.data_utils import get_diabetes_data
from support.models_base import IModel
from support.train_utils import train


class RandomForestRegressionModelWrapper(IModel):
    def __init__(self, regressor: RandomForestRegressor):
        self.model = regressor

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x) -> any:
        return self.model.predict(x)


def main():
    data_set = get_diabetes_data()

    print("Trees = 100, criterion = squared error")
    model = RandomForestRegressionModelWrapper(RandomForestRegressor())
    model_environment = RegressionModelEnvironment(model, data_set, random_state=1)
    train(model_environment)

    print("\n\nTrees = 10, criterion = squared error")
    model = RandomForestRegressionModelWrapper(RandomForestRegressor(n_estimators=10))
    model_environment = RegressionModelEnvironment(model, data_set, random_state=1)
    train(model_environment)

    print("\n\nTrees = 100, criterion = absolute error")
    model = RandomForestRegressionModelWrapper(RandomForestRegressor(criterion='absolute_error'))
    model_environment = RegressionModelEnvironment(model, data_set, random_state=1)
    train(model_environment)


if __name__ == '__main__':
    main()
