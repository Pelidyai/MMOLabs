from matplotlib import pyplot as plt
import seaborn as sms

from support.types import DataSet


def plot_iris_data_set(data_set: DataSet):
    colors = [{0: 'r', 1: 'g', 2: 'b'}[y] for y in data_set.y]
    plot_data_set(data_set, colors)


def plot_data_set(data_set: DataSet, colors: list = None):
    data_frame = data_set.as_data_frame()
    feature_names = data_set.feature_names
    figure, subplots = plt.subplots(len(feature_names), len(feature_names))
    for i, first_name in enumerate(feature_names):
        for j, second_name in enumerate(feature_names):
            subplots[i][j].scatter(data_frame[first_name], data_frame[second_name], c=colors)
            subplots[i][j].set_xlabel(first_name)
            subplots[i][j].set_ylabel(second_name)
    plt.show()
    sms.heatmap(data_frame.corr(), annot=True)
    plt.show()

