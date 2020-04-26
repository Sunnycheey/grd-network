"""
Main module include function definition and corresponding implement

Note:
    * There are many distance between different feature, we simply using Euclidean distance here
    * We simply add the distances up (with different weight) to get over all distance between feature point
    * As for weight, we using the algorithm mentioned in paper to get optimised combination of weights
"""

from __future__ import annotations

import pandas as pd
from typing import List
import numpy as np
from sklearn import neighbors
from sklearn.metrics import pairwise_distances


class Core:
    def __init__(self, path: str) -> Core:
        self.model = None
        self.data = None
        self.path = path

    def load_data(self):
        f = open(self.path, 'r')
        df = pd.read_csv(f, sep=',', header=None)
        self.data = df.values
        f.close()

    def train_model(self):
        target = self.data[:, -1].astype(int)
        data = self.data[:, :-1]
        clf = neighbors.KNeighborsClassifier(metric=Core.get_distance)
        clf.fit(data, target)
        self.model = clf
        print('model trained successfully!')

    def util(self):
        self.load_data()
        self.train_model()

    def predict(self) -> np.ndarray:
        # return the predict result
        return self.model.predict(self.data[:,:-1])

    @staticmethod
    def get_features(x: np.ndarray) -> List[np.ndarray]:
        # you can find the detailed description of feature in the document
        # this function return a list of ndarray, the length of list is 6
        ret = []
        _feature_1 = x[0:4]
        # feature_2 and feature_3 are padding up to fix size (max length is 200)
        # todo: handle sparse vector
        _feature_2 = x[4:204]
        _feature_3 = x[204:404]
        _feature_4 = x[404]
        _feature_5 = x[405:408]
        _feature_6 = x[408:428]
        ret.append(_feature_1)
        ret.append(_feature_2)
        ret.append(_feature_3)
        ret.append(_feature_4)
        ret.append(_feature_5)
        ret.append(_feature_6)
        return ret

    @staticmethod
    def get_distance(x: np.ndarray, y: np.ndarray) -> float:
        """
        Function for measure the distance between feature point
        :param x: feature point x
        :param y: feature point y
        :return: the distance between them
        """
        x = Core.get_features(x)
        y = Core.get_features(y)
        distance = 0
        # x and y have the same shape
        for i in range(len(x)):
            # reshape the ndarray from 1 dim to 2 dim
            if i != 1 and i != 2:
                a = x[i].reshape(1, -1)
                b = y[i].reshape(1, -1)

                distance += pairwise_distances(a, b, metric='manhattan').item()
            else:
                # low of efficient :(
                a = list(reversed(x[i].tolist()))
                b = list(reversed(y[i].tolist()))
                idx_a = idx_b = 0
                for idx in range(len(a)):
                    if a[idx] != 0:
                        idx_a = idx
                        break
                for idx in range(len(b)):
                    if b[idx] != 0:
                        idx_b = idx
                        break
                if idx_a == idx_b:
                    a = np.asarray(a).reshape(1, -1)
                    b = np.asarray(b).reshape(1, -1)
                    distance += pairwise_distances(a, b, metric='manhattan').item()

                # if idx_x != idx_y the distance to be add is zero according to paper

        return distance


if __name__ == '__main__':

    core = Core('../../data/features.csv')
    core.util()
    print(core.predict())
