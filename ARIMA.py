# -*- coding: utf-8 -*-
"""
@Time : 2022/10/9 20:31 
@Author : Chao Zhu
"""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
from dataset import LoadData


config = {
    'model': 'ARIMA',
    'p_range': [0, 4],
    'd_range': [0, 3],
    'q_range': [0, 4],
    'input_window': 12,
    'output_window': 1,
}


# Try to find the best (p,d,q) parameters for ARIMA
def order_select_pred(data):
    # data: (T, F)
    res = ARIMA(data, order=(0, 0, 0)).fit()
    bic = res.bic
    p_range = config.get('p_range', [0, 4])
    d_range = config.get('d_range', [0, 3])
    q_range = config.get('q_range', [0, 4])
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=ConvergenceWarning)
        warnings.simplefilter("error", category=RuntimeWarning)
        for p in range(p_range[0], p_range[1]):
            for d in range(d_range[0], d_range[1]):
                for q in range(q_range[0], q_range[1]):
                    try:
                        cur_res = ARIMA(data, order=(p, d, q)).fit()
                    except:
                        continue
                    if cur_res.bic < bic:
                        bic = cur_res.bic
                        res = cur_res
    return res


def arima(data):  # (num_samples, in, F)
    output_window = config.get('output_window', 1)
    y_pred = []  # (num_samples, out, F)
    for time_slot in tqdm(data, 'ts'):  # (in, F)
        # Different nodes should be predict by different ARIMA models instance.
        pred = order_select_pred(time_slot).forecast(steps=output_window)
        pred = pred.reshape((-1, time_slot.shape[1]))  # (out, F)
        print(pred)
        y_pred.append(pred)
    return np.array(y_pred)  # (num_samples, out, F)


def main():
    print(config)
    train_data = LoadData(data_path='../data/transaction_send_rates/ReserveSendRates_5minutes.npy', divide_days=[24, 6],
                          time_interval=5, history_length=12, train_mode='train')
    test_data = LoadData(data_path='../data/transaction_send_rates/ReserveSendRates_5minutes.npy', divide_days=[24, 6],
                         time_interval=5, history_length=12, train_mode='test')
    testx = test_data[0]['park_x'].unsqueeze(0).numpy()  # (1, 12, 1)
    testy = test_data[0]['park_y'].unsqueeze(0).numpy()  # (1, 1, 1)
    for i in range(1, len(test_data)):  # 1728
        single_test_data = test_data[i]
        testx = np.vstack((testx, single_test_data['park_x'].unsqueeze(0).numpy()))  # (1728, 12, 1)
        testy = np.vstack((testy, single_test_data['park_y'].unsqueeze(0).numpy()))  # (1728, 1, 1)
    y_pred = arima(testx)
    # y_true = pd.Series(np.load('../result/y_true_arima.npy').reshape(1728, ), name='y_true')
    # y_pred = pd.Series(np.load('../result/y_pred_arima.npy').reshape(1728, ), name='y_pred')
    #
    # y = pd.concat([y_true, y_pred], axis=1)
    # y.to_csv('../result/y_true_pred_arima.csv')
    # print(y)


if __name__ == '__main__':
    main()
