import numpy as np
import pandas as pd


class DataSet:
    def __init__(self, x: np.ndarray, y: np.ndarray, feature_names):
        if len(x) != len(y):
            raise Exception("Size of X series should be equal to Y series")
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.feature_names = feature_names

    def as_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.x, columns=self.feature_names)
