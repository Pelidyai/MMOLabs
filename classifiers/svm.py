from matplotlib import pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

from classifiers.base import ClassifierModelEnvironment
from support.data_utils import get_iris_data
from support.models_base import IModel
from support.plot_utils import plot_iris_data_set
from support.train_utils import train


class SVMModelWrapper(IModel):
    def __init__(self, svm: SVC):
        self.model = svm

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x) -> any:
        return self.model.predict(x)


def main(should_plot=False):
    data_set = get_iris_data()
    data_set.x = data_set.x[:, :2]
    if should_plot:
        plot_iris_data_set(data_set)

    models = []

    print("Kernel - rbf")
    model = SVMModelWrapper(SVC(kernel='rbf'))
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)
    models.append(model.model)

    print("\n\nKernel - linear")
    model = SVMModelWrapper(SVC(kernel='linear'))
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)
    models.append(model.model)

    print("\n\nKernel - poly")
    model = SVMModelWrapper(SVC(kernel='poly'))
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)
    models.append(model.model)

    print("\n\nKernel - sigmoid")
    model = SVMModelWrapper(SVC(kernel='sigmoid'))
    model_environment = ClassifierModelEnvironment(model, data_set, random_state=1)
    train(model_environment)
    models.append(model.model)

    titles = (
        "SVC with RBF kernel",
        "SVC with linear kernel",
        "SVC with polynomial (degree 3) kernel",
        "SVC with sigmoid kernel",
    )

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = data_set.x[:, 0], data_set.x[:, 1]

    for clf, title, ax in zip(models, titles, sub.flatten()):
        disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            data_set.x,
            response_method="predict",
            cmap=plt.cm.coolwarm,
            alpha=0.8,
            ax=ax,
            xlabel=data_set.feature_names[0],
            ylabel=data_set.feature_names[1],
        )
        ax.scatter(X0, X1, c=data_set.y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()


if __name__ == '__main__':
    main()
