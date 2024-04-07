import tensorflow as tf
from tensorflow.python.keras import activations

from regressors.base import RegressionModelEnvironment
from support.data_utils import get_diabetes_data
from support.models_base import IModel
from support.train_utils import train


class GoodRegressorNN(tf.keras.Sequential):
    def __init__(self):
        super(GoodRegressorNN, self).__init__(name='')
        self.add(tf.keras.layers.Dense(240, activation=activations.leaky_relu))
        self.add(tf.keras.layers.Dropout(rate=0.1))

        self.add(tf.keras.layers.Dense(90, activation=activations.softplus))
        self.add(tf.keras.layers.Dropout(rate=0.05))

        self.add(tf.keras.layers.Dense(1, activation=activations.linear))


class BadRegressorNN(tf.keras.Sequential):
    def __init__(self):
        super(BadRegressorNN, self).__init__(name='')
        self.add(tf.keras.layers.Dense(3, activation=activations.leaky_relu))
        self.add(tf.keras.layers.Dropout(rate=0.1))

        self.add(tf.keras.layers.Dense(2, activation=activations.softplus))
        self.add(tf.keras.layers.Dropout(rate=0.05))

        self.add(tf.keras.layers.Dense(1, activation=activations.linear))


class NNWrapper(IModel):
    def __init__(self, nn_model: tf.keras.Sequential):
        self.nn = nn_model
        self.nn.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.mean_squared_error)

    def fit(self, x, y):
        self.nn.fit(x, y, verbose=0, epochs=100)

    def predict(self, x) -> any:
        return self.nn.predict(x)


def main():
    data_set = get_diabetes_data()

    print("Good regressor: 10x240x90x1")
    model = NNWrapper(GoodRegressorNN())
    model_environment = RegressionModelEnvironment(model, data_set, random_state=1)
    train(model_environment)

    print("Bad regressor: 10x15x5x1")
    model = NNWrapper(BadRegressorNN())
    model_environment = RegressionModelEnvironment(model, data_set, random_state=1)
    train(model_environment)


if __name__ == '__main__':
    main()
