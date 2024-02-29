from sklearn.datasets import load_iris, load_breast_cancer

from support.types import DataSet


def get_iris_data() -> DataSet:
    data = load_iris()
    return DataSet(data['data'], data['target'], data['feature_names'])


def get_breast_cancer_data() -> DataSet:
    data = load_breast_cancer()
    return DataSet(data['data'], data['target'], data['feature_names'])
