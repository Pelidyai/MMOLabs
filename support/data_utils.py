from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes

from support.types import DataSet


def get_iris_data() -> DataSet:
    data = load_iris()
    return DataSet(data['data'], data['target'], data['feature_names'])


def get_breast_cancer_data() -> DataSet:
    data = load_breast_cancer()
    return DataSet(data['data'], data['target'], data['feature_names'])


def get_diabetes_data() -> DataSet:
    data = load_diabetes()
    return DataSet(data['data'], data['target'], data['feature_names'])
