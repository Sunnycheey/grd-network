"""
Main module include function definition and corresponding implement

Note:
    * There are many distance between different feature, we simply using Manhattan distance here
    * We simply add the distances up (with different weight) to get over all distance between feature point
    * As for weight, we using the algorithm mentioned in paper to get optimised combination of weights
"""

from __future__ import annotations

import pandas as pd
from typing import List
import numpy as np
from sklearn import neighbors
from sklearn.metrics import pairwise_distances
import json


class Core:
    def __init__(self, path: str, optimizer_flag: bool, weights_path) -> Core:
        self.model = None
        self.data = None
        self.path = path
        self.x = None
        self.y = None
        if not optimizer_flag:
            # self.weights = np.asarray([2.891, 0.922, 0.880, 0.040, 1.277, 0.898])
            with open(weights_path, 'r') as f:
                obj = json.load(f)
            self.weights = np.asarray(obj['weight'])
        else:
            self.weights = np.random.uniform(0.5, 1.5, (1, 6)).flatten()

    def load_data(self):
        f = open(self.path, 'r')
        df = pd.read_csv(f, sep=',', header=None)
        self.data = df.values
        f.close()

    def train_model(self):
        target = self.data[:, -1].astype(int)
        self.y = target
        data = self.data[:, :-1]
        self.x = data
        clf = neighbors.KNeighborsClassifier(metric=self.get_distance)
        clf.fit(data, target)
        self.model = clf
        print('model trained successfully!')

    def util(self):
        self.load_data()
        self.train_model()
        # print(self.weights)

    def predict(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        with open(path, 'r') as f:
            df = pd.read_csv(f, sep=',', header=None)
        val = df.values
        ground_truth = val[:, -1]
        # return the predict result
        return self.model.predict(val[:, :-1]), ground_truth

    def eval(self, path: str) -> float:
        result, ground_truth = self.predict(path)
        print('predict result: {}'.format(result.tolist()))
        diff = result - ground_truth
        acc = np.sum(diff == 0) / ground_truth.size
        return acc

    def optimizer(self, R: int = 1):
        """
        Find the optimised combination of weights
        :return:
        """
        """
        R is the rounds that repeat the training set R times
        kreco is the closest k points of each point
        P_train is the current point of train
        S is the other point of train
        """
        for r in range(R):
            kreco = 5
            for i in range(len(self.y)):
                S_good = []
                S_bad = []
                P_train_x = self.x[i, :]
                P_train_y = self.y[i]
                """
                Find the S_good
                S_good[0:k] is the distance of closest k points and the same class of Current point
                """
                for j in range(len(self.y)):
                    S_x = self.x[j, :]
                    S_y = self.y[j]
                    if P_train_y == S_y and i != j:
                        S_good.append(Core.get_distance(self, P_train_x, S_x))
                    else:
                        S_good.append(float("inf"))
                S_good_sort = np.argsort(S_good)
                # print(S_good_sort[0:5])
                """
                Find the d_maxgood_feature
                d_maxgood_feature is the set of max distance (Compared with point in S_good[0:k]) in every feature.
                """
                d_maxgood = []
                for feature in range(6):
                    max = 0
                    for k in range(kreco):
                        a = Core.get_distance_i(P_train_x, self.x[S_good_sort[k], :], feature)
                        # print(a)
                        if Core.get_distance_i(P_train_x, self.x[S_good_sort[k], :], feature) > max:
                            max = Core.get_distance_i(P_train_x, self.x[S_good_sort[k], :], feature)
                    d_maxgood.append(max)
                # print(d_maxgood)
                """
                Find the S_bad
                S_bad[0:k] is the distance of closest k points and the different class of Current point
                """
                for j in range(len(self.y)):
                    S_x = self.x[j, :]
                    S_y = self.y[j]
                    if P_train_y != S_y and i != j:
                        S_bad.append(Core.get_distance(self, P_train_x, S_x))
                    else:
                        S_bad.append(float("inf"))
                S_bad_sort = np.argsort(S_bad)
                """
                Find the n_bad
                n_bad is the number set that smaller than the distance of d_maxgood_feature in every feature.
                """
                n_bad = []
                for feature in range(6):
                    count = 0
                    for k in range(kreco):
                        if Core.get_distance_i(P_train_x, self.x[S_bad_sort[k], :], feature) <= d_maxgood[feature]:
                            count += 1
                    n_bad.append(count)
                print(n_bad)
                """
                Weight adjustment.
                n_bad_min is the smallest distance in all feature
                decrease_with_weight is the sum of all decrease of difference before and after with weights
                increase is the sum of increase without weights
                increase_feature[] is the feature that need increase weights
                """
                n_bad_sort = np.argsort(n_bad)
                n_bad_min = n_bad[n_bad_sort[0]]
                decrease = 0
                # increase = 0
                increase_feature = []
                for feature in range(6):
                    if n_bad[feature] > n_bad_min:
                        decrease += 0.01 * self.weights[feature] * n_bad[feature] / 5
                        self.weights[feature] = self.weights[feature] - 0.01 * self.weights[feature] * n_bad[
                            feature] / 5
                    else:
                        increase_feature.append(feature)
                every_increase = decrease / len(increase_feature)
                for l in range(len(increase_feature)):
                    self.weights[increase_feature[l]] = self.weights[increase_feature[l]] + every_increase
                print(self.weights)
                # print(self.weights.sum())

    def save(self, path: str):
        import json

        obj = {'weight': self.weights.tolist()}
        with open('weights.json', 'w') as f:
            json.dump(obj, f)

    def draw(self):
        if self.model is None:
            raise ValueError('you need train the model first')
        from sklearn.manifold.t_sne import TSNE
        import matplotlib.pyplot as plt

        X_Train_embedded = TSNE(n_components=2).fit_transform(self.x)
        y_predicted = self.model.predict(self.x)

        # replace the above by your data and model
        # create meshgrid
        resolution = 100  # 100x100 background pixels
        background_model = neighbors.KNeighborsClassifier(n_neighbors=1).fit(X_Train_embedded, y_predicted)
        X2d_xmin, X2d_xmax = np.min(X_Train_embedded[:, 0]), np.max(X_Train_embedded[:, 0])
        X2d_ymin, X2d_ymax = np.min(X_Train_embedded[:, 1]), np.max(X_Train_embedded[:, 1])
        xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))

        # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
        voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
        voronoiBackground = voronoiBackground.reshape((resolution, resolution))

        # plot
        plt.contourf(xx, yy, voronoiBackground)
        plt.scatter(X_Train_embedded[:, 0], X_Train_embedded[:, 1], c=self.y)
        plt.savefig('fig.png')
        # plt.show()

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

    def get_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Function for measure the distance between feature point
        :param weights: weights array
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
            val = 0
            if i != 1 and i != 2:

                val = Core.get_common_feature_distance(x[i], y[i])
                distance += val * self.weights[i]

            else:
                val = Core.get_padding_feature_distance(x[i], y[i])
                distance += val * self.weights[i]
            # print('distance of dim {} is {}'.format(i, val))

            # if idx_x != idx_y the distance to be add is zero according to paper

        return distance

    @staticmethod
    def get_common_feature_distance(x: np.ndarray, y: np.ndarray) -> int:
        a = x.reshape(1, -1)
        b = y.reshape(1, -1)
        val = pairwise_distances(a, b, metric='manhattan').item()
        return val

    @staticmethod
    def get_padding_feature_distance(x: np.ndarray, y: np.ndarray):
        val = 0
        a = x[::-1]
        b = y[::-1]
        # find position of the first occurrence of non -1 element
        idx_a = np.argmax(a != -1)
        idx_b = np.argmax(b != -1)

        if idx_a == idx_b:
            a = a[idx_a:].reshape(1, -1)
            b = b[idx_b:].reshape(1, -1)
            val = pairwise_distances(a, b, metric='manhattan').item()
        return val

    @staticmethod
    def get_distance_i(x: np.ndarray, y: np.ndarray, i) -> float:
        x = Core.get_features(x)
        y = Core.get_features(y)
        if i != 1 and i != 2:
            return Core.get_common_feature_distance(x[i], y[i])
        else:
            return Core.get_padding_feature_distance(x[i], y[i])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="classifier for wf based on handcraft feature based on knn")
    parser.add_argument('--do_eval', action='store_true', help='eval the model')
    parser.add_argument('--do_optimize', action='store_true', help='optimize weights')
    parser.add_argument('--ratio', type=int, default=80, help='set ratio (which dataset to load)')
    parser.add_argument('--weights_path', type=str, default='src/model/weights.json', help='indicate where to load '
                                                                                           'weight')

    args = parser.parse_args()

    if args.do_eval:
        core = Core('data/train_{}.csv'.format(args.ratio), False, args.weights_path)
        core.util()
        print(core.eval('data/test_{}.csv'.format(args.ratio)))

    if args.do_optimize:
        core = Core('data/train_{}.csv'.format(args.ratio), True, args.weights_path)
        core.util()
        core.optimizer()
        core.save(args.weights_path)
        print('result weights: {}'.format(core.weights.tolist()))
