from sklearn.datasets import load_iris

from support.types import DataSet


def get_iris_data() -> DataSet:
    data = load_iris()
    return DataSet(data['data'], data['target'], data['feature_names'])
