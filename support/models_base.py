from abc import abstractmethod

from sklearn.model_selection import train_test_split

from support.types import DataSet


class ModelEnvironment:
    def __init__(self, model, data_set: DataSet, test_size=0.2, shuffle=True, random_state=None):
        self._model = model
        (self._x_train, self._x_test,
         self._y_train, self._y_test) = train_test_split(data_set.x, data_set.y, test_size=test_size,
                                                         shuffle=shuffle, random_state=random_state)

    def fit(self):
        self._model.fit(self._x_train, self._y_train)

    @abstractmethod
    def score(self):
        pass

    @abstractmethod
    def get_score_info(self) -> str:
        pass
