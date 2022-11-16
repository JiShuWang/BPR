# -*- coding: utf-8 -*-
"""
@Time : 2022/1/7 8:54 
@Author : Chao Zhu
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import LoadData
from RNN import RNN
import numpy as np
import math
import random
import matplotlib.pyplot as plt


# set random seed
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


def MAE(y_true, y_pred):
    y_true = y_true.detach().numpy().copy().reshape((-1, 1))
    y_pred = y_pred.detach().numpy().copy().reshape((-1, 1))
    re = np.abs(y_true - y_pred).mean()

    return re


def RMSE(y_true, y_pred):
    y_true = y_true.detach().numpy().copy().reshape((-1, 1))
    y_pred = y_pred.detach().numpy().copy().reshape((-1, 1))
    re = math.sqrt(((y_true - y_pred) ** 2).mean())

    return re


def MAPE(y_true, y_pred):
    y_true = y_true.detach().numpy().copy().reshape((-1, 1))
    y_pred = y_pred.detach().numpy().copy().reshape((-1, 1))
    e = (y_true + y_pred) / 2 + 0.01
    re = (np.abs(y_true - y_pred) / (np.abs(y_true) + e)).mean()

    return re


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    # Loading Dataset
    train_data = LoadData(data_path='../data/transaction_send_rates/ReserveSendRates_5minutes.npy', divide_days=[24, 6],
                          time_interval=5, history_length=12, train_mode='train')
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=32)

    test_data = LoadData(data_path='../data/transaction_send_rates/ReserveSendRates_5minutes.npy', divide_days=[24, 6],
                         time_interval=5, history_length=12, train_mode='test')
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=32)

    # Loading Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_net = RNN(device=device, rnn_type='LSTM', input_size=1, hidden_size=128,
                 num_layers=2, dropout=0, bidirectional=True).to(device)
    criterion = nn.MSELoss()

    # Train Model
    my_net.train()
    lr = 0.001
    optimizer = optim.Adam(params=my_net.parameters(), lr=lr)
    Adam_epochs = 50
    for epoch in range(Adam_epochs):
        epoch_loss = 0.0
        epoch_mae = 0.0
        epoch_rmse = 0.0
        epoch_mape = 0.0
        num = 0
        start_time = time.time()
        for data in train_loader:  # {"park_x": (B, T, D), "park_y": (B, P, D)}
            optimizer.zero_grad()
            predict_value = my_net(data, device).to(torch.device('cpu'))  # [0, 1] -> recover
            loss = criterion(predict_value, data['park_y'])
            epoch_loss += loss.item()
            epoch_mae += MAE(data['park_y'], predict_value)
            epoch_rmse += RMSE(data['park_y'], predict_value)
            epoch_mape += MAPE(data['park_y'], predict_value)
            loss.backward()
            optimizer.step()
            num += 1
        end_time = time.time()
        epoch_mae = epoch_mae / num
        epoch_rmse = epoch_rmse / num
        epoch_mape = epoch_mape / num
        print('Epoch: {:04d}, Loss: {:02.4f}, mae: {:02.4f}, '
              'rmse: {:02.4f}, mape: {:02.4f}, Time: {:02.2f} mins'.format(epoch + 1,
                                                                           10 * epoch_loss / (len(train_data) / 64),
                                                                           epoch_mae,
                                                                           epoch_rmse,
                                                                           epoch_mape,
                                                                           (end_time - start_time) / 60))

    # Test Model
    my_net.eval()
    with torch.no_grad():
        epoch_mae = 0.0
        epoch_rmse = 0.0
        epoch_mape = 0.0
        total_loss = 0.0
        num = 0
        all_predict_value = 0
        all_y_true = 0
        for data in test_loader:
            predict_value = my_net(data, device).to(torch.device('cpu'))  # (B, P, D)
            if num == 0:
                all_predict_value = predict_value
                all_y_true = data['park_y']
            else:
                all_predict_value = torch.cat([all_predict_value, predict_value], dim=0)  # Concatenate the prediction values of all test data
                all_y_true = torch.cat([all_y_true, data['park_y']], dim=0)  # Concatenate the ground truth of all test data
            loss = criterion(predict_value, data['park_y'])
            total_loss += loss.item()
            num += 1
        epoch_mae = MAE(all_y_true, all_predict_value)
        epoch_rmse = RMSE(all_y_true, all_predict_value)
        epoch_mape = MAPE(all_y_true, all_predict_value)
        print('Test Loss: {:02.4f}, mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}'.format(10 * total_loss
                                                                                          / (len(test_data) / 64),
                                                                                          epoch_mae,
                                                                                          epoch_rmse,
                                                                                          epoch_mape))

    # save model parameters
    y_true = test_data.recover_data(test_data.park_norm[0], test_data.park_norm[1], all_y_true).numpy().reshape(1728,)
    y_pred = test_data.recover_data(test_data.park_norm[0], test_data.park_norm[1], all_predict_value).numpy().reshape(1728,)
    np.save("../result/y_true.npy", y_true)
    np.save("../result/y_pred.npy", y_pred)

    title = my_net.rnn_type + '_lr' + str(lr) + '_hs' + str(my_net.hidden_size) \
            + '_nl' + str(my_net.layers) + '_do' + str(my_net.dropout)
    torch.save(my_net, '../pth/' + title + '.pth')

    # Select time period to visualize
    plt.ylabel('transaction arrival rate')
    xticks = list(range(0, 289, 36))
    xticklabels = [str(i) + ':00' for i in range(0, 24, 3)] + ['00:00']
    # xticklabels = pd.date_range(start='2014-9-1 00:00:00', end='2014-9-2 00:00:00', freq='180min')
    plt.xticks(xticks, xticklabels, rotation=30)
    plt.plot(y_true[: 24 * 12], label='ground truth')
    plt.plot(y_pred[: 24 * 12], label='predict value')
    plt.legend()
    plt.savefig('../imgeva/1 day_' + title + '.png', dpi=400)
    plt.show()

    plt.ylabel('transaction arrival rate')
    xticks = list(range(0, 1729, 288))
    xticklabels = ['09-' + str(25 + i) for i in range(0, 6)] + ['10-01']
    plt.xticks(xticks, xticklabels)
    plt.plot(y_true[: 24 * 12 * 6], label='ground truth')
    plt.plot(y_pred[: 24 * 12 * 6], label='predict value')
    plt.legend()
    plt.savefig('../imgeva/6 days_' + title + '.png', dpi=400)
    plt.show()

    mae = MAE(test_data.recover_data(test_data.park_norm[0], test_data.park_norm[1], all_y_true),
              test_data.recover_data(test_data.park_norm[0], test_data.park_norm[1], all_predict_value))
    rmse = RMSE(test_data.recover_data(test_data.park_norm[0], test_data.park_norm[1], all_y_true),
                test_data.recover_data(test_data.park_norm[0], test_data.park_norm[1], all_predict_value))
    mape = MAPE(test_data.recover_data(test_data.park_norm[0], test_data.park_norm[1], all_y_true),
                test_data.recover_data(test_data.park_norm[0], test_data.park_norm[1], all_predict_value))
    print("Accuracy metrics based on raw values  mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}".format(mae, rmse, mape))


if __name__ == '__main__':
    main()
