import numpy as np
import pandas as pd
import sqlite3
import time
import os
import datetime as dt
from pyts.image import GramianAngularField
from dataset import ImageData
from torch.utils.data import DataLoader


class DataProcessor():
    """
    Data extraction, process and feature construction for futures.
    """
    def __init__(self):
        self.train_window = 30
        self.batch_size = 64
        self.pred_interval = 10

        self.features = ['open', 'high', 'low', 'close', 'volume']
        self.coins = ['TA888', 'V888', 'RU888', 'RB888', 'C888', 'FU888', 'J888', 'L888']

        self.Database = 'Data.db'
    
    def get_data(self, start_datetime, end_datetime):
        # [feature, datetime, coin]
        n_features = len(self.features)
        n_coins = len(self.coins)
        data_dict = {}

        start = self.parse_datetime(start_datetime)
        end = self.parse_datetime(end_datetime)
        
        with sqlite3.connect(self.Database) as connection:
            cursor = connection.cursor()
            
            for coin in self.coins:
                temp = pd.read_sql_query(f"""
                                        SELECT date AS date_norm, open, high, low, close, volume
                                        FROM History
                                        WHERE date >= ?
                                        AND date <= ?
                                        AND coin == ?
                                        """, connection, params=[start, end] + [coin],
                                        parse_dates=['date_norm'], index_col='date_norm')\
                                        .sort_values(['date_norm'])
                
                data_dict[coin] = temp

        print("The keys of data_dict : ", data_dict.keys())

        # lable data with y
        data_dict = self._label_data(data_dict)

        data_mat, y_mat = self._split_data(data_dict)
        data_mat = self.apply_gaf(data_mat)

        print(data_mat.shape)
        print(y_mat.shape)
        # generate dataloader
        data_loader = DataLoader(ImageData(data_mat, y_mat), batch_size=self.batch_size)

        return data_loader

    # @staticmethod
    # def read_data():
    #     try:
    #         data_mat = np.load('data_mat.npy')
    #         y_mat = np.load('y.npy')
    #         return data_mat, y_mat
    #     except:
    #         raise FileExistsError('Data not found...')

    def _label_data(self, data_dict):
        for data in data_dict.values():
            data['return_' + str(self.pred_interval)] = (data['close'].shift(self.pred_interval) - data['close'] > 0).astype(int)
        return data_dict

    def _split_data(self, data_dict):
        n = sum([df.shape[0] - (self.train_window - 1) - 10 for df in data_dict.values()])
        print('Number of samples: ', n)

        data_mat = np.zeros((n, self.train_window, len(self.features)))
        y_mat = np.zeros((n,))
 
        idx = 0
        for coin in self.coins:
            print(coin, "spliting...")
            temp = data_dict[coin]
            for i in range(self.train_window - 1, temp.shape[0] - 10):
                data_mat[idx, :, :] = temp.iloc[i - (self.train_window - 1):i + 1, 0:5]
                y_mat[idx] = temp.iloc[i, -1]
                idx += 1
    
        return data_mat, y_mat

    def apply_gaf(self, data_mat):
        data_mat = np.transpose(data_mat, (0, 2, 1))
        transformer = GramianAngularField()

        n_samples = data_mat.shape[0]
        images = np.zeros((n_samples, 5, self.train_window, self.train_window))
        
        for i, item in enumerate(data_mat):
            images[i, :, :, :] = transformer.transform(item)

        return images

    # @staticmethod
    # def apply_nt(data_mat):
    #     return np.expand_dims(np.transpose(data_mat, (0, 2, 1)), axis=1)

    @staticmethod
    def parse_datetime(date_time):
        return time.mktime(dt.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S').timetuple())
