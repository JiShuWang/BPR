# -*- coding: utf-8 -*-
"""
@Time : 2022/10/10 9:55 
@Author : Chao Zhu
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from dataset import LoadData

config = {
    'model': 'SVR',
    'kernel': 'rbf',
    'input_window': 12,
    'output_window': 1,
}


def run_SVR():
    output_window = config.get("output_window", 1)
    kernel = config.get('kernel', 'rbf')

    y_pred = []
    y_true = []

    train_data = LoadData(data_path='../data/transaction_send_rates/ReserveSendRates_5minutes.npy',
                          divide_days=[24, 6],
                          time_interval=5, history_length=12, train_mode='train')
    test_data = LoadData(data_path='../data/transaction_send_rates/ReserveSendRates_5minutes.npy',
                         divide_days=[24, 6],
                         time_interval=5, history_length=12, train_mode='test')

    trainx = train_data[0]['park_x'].unsqueeze(0).numpy()  # (1, 12, 1)
    trainy = train_data[0]['park_y'].unsqueeze(0).numpy()  # (1, 1, 1)
    for i in range(1, len(train_data)):  # 6900
        single_train_data = train_data[i]
        trainx = np.vstack((trainx, single_train_data['park_x'].unsqueeze(0).numpy()))  # (6900, 12, 1)
        trainy = np.vstack((trainy, single_train_data['park_y'].unsqueeze(0).numpy()))  # (6900, 1, 1)

    testx = test_data[0]['park_x'].unsqueeze(0).numpy()  # (1, 12, 1)
    testy = test_data[0]['park_y'].unsqueeze(0).numpy()  # (1, 1, 1)
    for i in range(1, len(test_data)):  # 1728
        single_test_data = test_data[i]
        testx = np.vstack((testx, single_test_data['park_x'].unsqueeze(0).numpy()))  # (1728, 12, 1)
        testy = np.vstack((testy, single_test_data['park_y'].unsqueeze(0).numpy()))  # (1728, 1, 1)
    # (train_size, in/out, F), (test_size, in/out, F)

    trainx = np.reshape(trainx, (trainx.shape[0], -1))  # (train_size, in * F)
    trainy = np.reshape(trainy, (trainy.shape[0], -1))  # (train_size, out * F)
    # trainy = np.mean(trainy, axis=1)  # (train_size,)
    testx = np.reshape(testx, (testx.shape[0], -1))  # (test_size, in * F)
    print(trainx.shape, trainy.shape, testx.shape, testy.shape)

    svr_model = SVR(kernel=kernel, C=1e3, gamma=1e-8, epsilon=0.1)
    svr_model.fit(trainx, trainy)
    pre = svr_model.predict(testx)  # (test_size, )
    pre = np.expand_dims(pre, axis=1)  # (test_size, 1)
    f = 1
    pre = pre.repeat(output_window * f, axis=1)  # (test_size, out * F)
    y_pred.append(pre.reshape(pre.shape[0], output_window, f))  # (test_size, out, F)
    y_true.append(testy)  # (test_size, out, F)

    y_pred = np.array(y_pred)  # (test_size, out, F)
    y_true = np.array(y_true)  # (test_size, out, F)

    return y_pred, y_true


def main():
    print(config)
    y_pred, y_true = run_SVR()
    print(y_pred.shape)
    print(y_true.shape)
    y_true = pd.Series(y_true.reshape(1728, ), name='y_true')
    y_pred = pd.Series(y_pred.reshape(1728, ), name='y_pred')

    y = pd.concat([y_true, y_pred], axis=1)
    y.to_csv('../result/y_true_pred_svr.csv')


if __name__ == '__main__':
    main()
