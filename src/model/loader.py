"""
Module for loading data (in batch)
"""

from __future__ import annotations

from itertools import islice


class DataLoader:
    """
    class for load feature dataset
    """

    def __init__(self, file_path: str, start_pos: int = 0, batch_size: int = 32) -> DataLoader:
        self.path = file_path
        self.start_pos = start_pos
        self.batch_size = batch_size
        self.current_pos = start_pos
        self.stop_iteration = False

    def __iter__(self):
        return self

    def __next__(self):
        f = open(self.path, 'r')
        if self.stop_iteration:
            f.close()
            raise StopIteration
        batch = islice(f, self.current_pos, self.current_pos + self.batch_size)
        self.current_pos += self.batch_size
        ret = [(idx, val) for (idx, val) in enumerate(batch)]
        if 0 < len(ret) < self.batch_size:
            self.stop_iteration = True
        elif len(ret) == 0:
            f.close()
            raise StopIteration
        return ret


if __name__ == '__main__':

    dataLoader = DataLoader('../../data/features.csv')
    break_flag = False
    iter_number = 0
    total_number = 0
    for data in dataLoader:
        iter_number += 1
        total_number += len(data)
        for (idx, val) in data:
            w = idx + 1
    print(iter_number, total_number)