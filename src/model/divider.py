"""
This module is used to divide original dataset to train set and test set
"""


class Divider:
    def __init__(self, path):
        self.path = path
        # the ratio is 80/20
        self.ratio = 80

    def divide(self):
        import pandas as pd
        import numpy as np
        import os
        train_set = np.empty((int(5000*self.ratio/100), 429))
        test_set = np.empty((int(5000*self.ratio/100), 429))
        train_idx = 0
        test_idx = 0
        with open(self.path, 'r') as f:
            df = pd.read_csv(f, sep=',', header=None)
        val = df.values
        # for reproducible, set seed
        np.random.seed(7)
        np.random.shuffle(val)
        for i in range(50):
            train_idx = i * self.ratio
            test_idx = i * (100 - self.ratio)
            samples = val[val[:, -1] == i]
            try:
                train_set[train_idx: train_idx + self.ratio] = samples[0:self.ratio]
                test_set[test_idx: test_idx + 100 - self.ratio] = samples[self.ratio: 100]
            except ValueError as e:
                print('Error: {}, i: {}'.format(e, i))

        # write the result to csv file, namely train.csv and test.csv
        dir = os.path.dirname(self.path)
        np.savetxt(os.path.join(dir, 'train.csv'), train_set, delimiter=',')
        np.savetxt(os.path.join(dir, 'test.csv'), test_set, delimiter=',')


if __name__ == '__main__':
    divider = Divider('../../data/features.csv')
    divider.divide()
