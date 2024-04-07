from typing import Callable, Any

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from support.models_base import ModelEnvironment, IModel
from support.types import DataSet


class RegressionScore:
    def calc_regression_error(self, x, y_target, error_func: Callable[[list, list], float]) -> float:
        y_predicted = self.predict(x)
        return error_func(y_target, y_predicted)

    def __init__(self, model: IModel = None, x: np.ndarray = None, y: np.ndarray = None):
        self.__model = model
        self.accuracy = None
        self.precision = None
        self.recall = None
        if model is None or x is None or y is None:
            return

        self.mae = self.calc_regression_error(x, y, mean_absolute_error)
        self.mape = self.calc_regression_error(x, y, mean_absolute_percentage_error)
        self.mse = self.calc_regression_error(x, y, mean_squared_error)

    def __str__(self):
        if self.mae is None:
            return "Not calculated"
        return (f'MAE - {round(self.mae, 3)}, '
                f'MAPE - {round(self.mape, 3)}, '
                f'MSE - {round(self.mse, 3)}, ')

    def predict(self, x) -> Any:
        return self.__model.predict(x)


class RegressionModelEnvironment(ModelEnvironment):
    def __init__(self, model: IModel, data_set: DataSet, test_size=0.2, shuffle=True, random_state=None):
        super().__init__(model, data_set, test_size, shuffle, random_state)
        self.train_score: RegressionScore = RegressionScore()
        self.test_score: RegressionScore = RegressionScore()

    def score(self):
        self.train_score = RegressionScore(self._model, self._x_train, self._y_train)
        self.test_score = RegressionScore(self._model, self._x_test, self._y_test)

    def get_score_info(self) -> str:
        return (f'Train score: {self.train_score}\n'
                f'Test score: {self.test_score}')
