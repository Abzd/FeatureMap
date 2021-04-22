import numpy as np
import pandas as pd
import time
import os
import pickle
import datetime as dt
from pyts.image import GramianAngularField
from dataset import ImageData
from torch.utils.data import DataLoader


class IndexProcessor:
    """
    Data extraction, process and feature construction for futures.
    """
    def __init__(self):
        self.train_portion = 0.8
        self.cv_portion = 0.1
        self.n_features = 5
        self.predict_days = 10
        self.train_window = 30

    def _read_from_file(self, root_path='data/factoradj/'):
        data_all = []
        for file in os.listdir(root_path):
            with open(root_path + file, 'rb') as f:
                if file != '.DS_Store':
                    temp = pickle.load(f) 
                    data_all.append(temp)
                    if file == 'Close.pkl':
                        # 1 day change
                        y_mat = (temp.shift(self.predict_days) - temp > 0).astype(int)
        
        n_days, n_index = data_all[0].shape
        data_mat = np.zeros((n_days, self.n_features, n_index))
        for idx, df in enumerate(data_all):
            data_mat[:, idx, :] = df

        return data_mat, np.array(y_mat)

    def get_data(self, root_path='data/factoradj/'):
        data_mat, y_mat = self._read_from_file(root_path)
        n_days, _, n_index= data_mat.shape

        data, y = [], []
        for date in range(self.train_window - 1, n_days - self.predict_days):
            for index in range(n_index):
                temp = data_mat[date - (self.train_window - 1):date + 1, :, index].T
                y_temp = y_mat[date, index]
                if not np.isnan(temp).any() and not np.isnan(y_temp).any():
                    data.append(temp)
                    y.append(y_temp)

        return np.array(data), np.array(y)             
         
    def apply_gaf(self, data):
        if os.path.exists('data/gaf_images.npy'):
            return np.load('data/gaf_images.npy')

        transformer = GramianAngularField()

        n_samples = data.shape[0]
        images = np.zeros((n_samples, 5, self.train_window, self.train_window))
        
        for i, item in enumerate(data):
            images[i, :, :, :] = transformer.transform(item)

        np.save('data/gaf_images.npy', arr=images)
        return images

    def split_data(self, images, y):
        n_train = int(images.shape[0] * self.train_portion)
        n_cv = int(images.shape[0] * self.cv_portion)
        x_train, x_cv, x_test = images[:n_train], images[n_train:(n_train + n_cv)], images[(n_train + n_cv):]
        y_train, y_cv, y_test = y[:n_train], y[n_train:(n_train + n_cv)], y[(n_train + n_cv):]
        
        train = ImageData(x_train, y_train)
        cv = ImageData(x_cv, y_cv)
        test = ImageData(x_test, y_test)

        train_loader = DataLoader(train, batch_size=64)
        cv_loader = DataLoader(cv, batch_size=64)
        test_loader = DataLoader(test, batch_size=64)
        return train_loader, cv_loader, test_loader

    @staticmethod
    def apply_nt(data):
        # for idx, item in enumerate(data):
        #     data[idx] = (item - np.min(item, axis=1).reshape((5, 1))) / (np.max(item, axis=1) - np.min(item, axis=1)).reshape((5, 1))
        return np.expand_dims(data, axis=1)
        
