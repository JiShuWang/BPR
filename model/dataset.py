# -*- coding: utf-8 -*-
"""
@Time : 2022/1/6 20:43 
@Author : Chao Zhu
"""
import torch
import numpy as np
from torch.utils.data import Dataset


class LoadData(Dataset):
    def __init__(self, data_path, divide_days, time_interval, history_length, train_mode):
        """
        :param data_path: list, "Number of parking data file name", path to save the data file names.
        :param divide_days: list, [days of train data, days of test data], list to divide the original data.
        :param time_interval: int, time interval between two  data records (mins).
        :param history_length: int, length of history data to be used.
        :param train_mode: list, ["train", "test"].
        """
        self.data_path = data_path
        self.train_days = divide_days[0]
        self.test_days = divide_days[1]
        self.time_interval = time_interval
        self.history_length = history_length
        self.train_mode = train_mode

        self.one_day_length = int(24 * 60 / self.time_interval)

        self.park_norm, self.park_data = self.preprocess_data(data=LoadData.get_park_data(park_file=self.data_path),
                                                              norm_dim=0)

    def __len__(self):
        """
        :return: length of dataset (number of samples).
        """
        if self.train_mode == 'train':
            return self.train_days * self.one_day_length - self.history_length
        elif self.train_mode == 'test':
            return self.test_days * self.one_day_length
        else:
            return ValueError('train mode: [{}] is not defined'.format(self.train_mode))

    def __getitem__(self, index):
        """
        :param index: int, range between [0, length-1].
        :return:
            data_x: torch.tensor, [T=12, D=1].
            data_y: torch.tensor, [P=1, D=1].
        """
        if self.train_mode == 'train':
            index = index
        elif self.train_mode == 'test':
            index += self.train_days * self.one_day_length
        else:
            raise ValueError('train mode: [{}] is not defined'.format(self.train_mode))

        data_x, data_y = LoadData.slice_data(self.park_data, self.history_length, index, self.train_mode)
        data_x = LoadData.to_tensor(data_x)
        data_y = LoadData.to_tensor(data_y).unsqueeze(1)

        return {'park_x': data_x, 'park_y': data_y}

    @staticmethod
    def get_park_data(park_file):
        """
        :param park_file: str, path of .npy file to save the parking data
        :return:
            np.array(T=52704, D=1)
        """
        park_data = np.load(park_file)

        return park_data

    @staticmethod
    def slice_data(data, history_length, index, train_mode):
        """
        :param data: np.array, normalized traffic data.
        :param history_length: int, length of history data to be used.
        :param pred_length: int, length of data to be predicted.
        :param index: int, index on temporal axis.
        :param train_mode: str, ["train", "test"].
        :return:
            data_x: np.array, [H, D].
            data_y: np.array, [P, D].
        """
        if train_mode == 'train':
            start_index = index
            end_index = index + history_length
        elif train_mode == 'test':
            start_index = index - history_length
            end_index = index
        else:
            raise ValueError('train mode [{}] is not defined'.format(train_mode))

        data_x = data[start_index: end_index, :]
        data_y = data[end_index, :]

        return data_x, data_y

    @staticmethod
    def preprocess_data(data, norm_dim):
        """
        :param data: np.array, original park data without normalization.
        :param norm_dim: int, normalization dimension.
        :return:
            norm_base: tuple, (max_data, min_data), data of normalization base.
            norm_data: np.array, max-min normalized park data.
        """
        norm_base = LoadData.normalize_base(data, norm_dim)
        norm_data = LoadData.normalize_data(norm_base[0], norm_base[1], data)

        return norm_base, norm_data

    @staticmethod
    def normalize_base(data, norm_dim):
        """
        :param data: np.array, original park data without normalization.
        :param norm_dim: int, normalization dimension.
        :return:
            max_data: np.array
            min_data: np.array
        """
        max_data = np.max(data, norm_dim, keepdims=True)
        min_data = np.min(data, norm_dim, keepdims=True)

        return max_data, min_data

    @staticmethod
    def normalize_data(max_data, min_data, data):
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, original park data without normalization.
        :return:
            np.array, max-min normalized park data.
        """
        normalized_data = (data - min_data) / (max_data - min_data)

        return normalized_data

    @staticmethod
    def recover_data(max_data, min_data, data):
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, max-min normalized data.
        :return:
            recovered_data: np.array, recovered data.
        """
        recovered_data = data * (max_data - min_data) + min_data

        return recovered_data

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)


if __name__ == '__main__':
    train_data = LoadData(data_path='../data/transaction_send_rates/ReserveSendRates_5minutes.npy', divide_days=[24, 6],
                          time_interval=5, history_length=12, train_mode='train')
    print(len(train_data))
    print(train_data[0]['park_x'].size())
    print(train_data[0]['park_y'].size())

    test_data = LoadData(data_path='../data/transaction_send_rates/ReserveSendRates_5minutes.npy', divide_days=[24, 6],
                         time_interval=5, history_length=12, train_mode='test')
    print(len(test_data))
    print(test_data[0]['park_x'].size())
    print(test_data[0]['park_y'].size())
