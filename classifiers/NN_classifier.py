import numpy as np
from keras.utils import to_categorical

from classifiers.base import ClassifierModelEnvironment
from support.data_utils import get_iris_data
from support.models_base import IModel
from support.plot_utils import plot_iris_data_set
from support.train_utils import train

import tensorflow as tf

from tensorflow.python.keras import activations


class GoodClassifierNN(tf.keras.Sequential):
    def __init__(self):
        super(GoodClassifierNN, self).__init__(name='')
        self.add(tf.keras.layers.Dense(80, activation=activations.leaky_relu))
        self.add(tf.keras.layers.Dropout(rate=0.1))

        self.add(tf.keras.layers.Dense(30, activation=activations.sigmoid))
        self.add(tf.keras.layers.Dropout(rate=0.05))

        self.add(tf.keras.layers.Dense(3, activation=activations.softmax))


class BadClassifierNN(tf.keras.Sequential):
    def __init__(self):
        super(BadClassifierNN, self).__init__(name='')
        self.add(tf.keras.layers.Dense(1, activation=activations.leaky_relu))
        self.add(tf.keras.layers.Dropout(rate=0.1))

        self.add(tf.keras.layers.Dense(2, activation=activations.sigmoid))
        self.add(tf.keras.layers.Dropout(rate=0.05))

        self.add(tf.keras.layers.Dense(3, activation=activations.softmax))


class NNWrapper(IModel):
    def __init__(self, nn_model: tf.keras.Sequential):
        self.nn = nn_model
        self.nn.compile(tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, x, y):
        self.nn.fit(x, y, verbose=0, epochs=100)

    def predict(self, x) -> any:
        return np.argmax(self.nn.predict(x), axis=-1)


def main(should_plot=False):
    data_set = get_iris_data()
    data_set.y = to_categorical(data_set.y)
    if should_plot:
        plot_iris_data_set(data_set)

    print("Good classifier: 4x8x3x3")
    model = NNWrapper(GoodClassifierNN())
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)

    print("Bad classifier: 4x1x2x3")
    model = NNWrapper(BadClassifierNN())
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)


if __name__ == '__main__':
    main()
