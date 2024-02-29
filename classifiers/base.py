from typing import Callable, Any

from sklearn.metrics import accuracy_score, precision_score, recall_score

from support.models_base import ModelEnvironment
from support.types import DataSet


class ClassificationScore:
    def calc_classification_error(self, x, y_target, error_func: Callable[[list, list], float]) -> float:
        y_predicted = self.predict(x)
        return error_func(y_target, y_predicted)

    def __init__(self, model=None, x=None, y=None):
        self.__model = model
        self.accuracy = None
        self.precision = None
        self.recall = None
        if model is None or x is None or y is None:
            return
        self.accuracy = self.calc_classification_error(x, y, accuracy_score)
        self.precision = self.calc_classification_error(x, y,
                                                        lambda y_t, y_p: precision_score(y_t, y_p,
                                                                                         average='weighted'))
        self.recall = self.calc_classification_error(x, y,
                                                     lambda y_t, y_p: recall_score(y_t, y_p, average='weighted'))

    def __str__(self):
        if self.accuracy is None:
            return "Not calculated"
        return (f'Accuracy - {round(self.accuracy, 3)}, '
                f'Precision - {round(self.precision, 3)}, '
                f'Recall - {round(self.recall, 3)}')

    def predict(self, x) -> Any:
        return self.__model.predict(x)


class ClassifierModelEnvironment(ModelEnvironment):
    def __init__(self, model, data_set: DataSet, test_size=0.2, shuffle=True, random_state=None):
        super().__init__(model, data_set, test_size, shuffle, random_state)
        self.train_score: ClassificationScore = ClassificationScore()
        self.test_score: ClassificationScore = ClassificationScore()

    def score(self):
        self.train_score = ClassificationScore(self._model, self._x_train, self._y_train)
        self.test_score = ClassificationScore(self._model, self._x_test, self._y_test)

    def get_score_info(self) -> str:
        return (f'Train score: {self.train_score}\n'
                f'Test score: {self.test_score}')
