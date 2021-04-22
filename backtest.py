import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import time
import datetime as dt
import sqlite3
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField


class Backtest:
    def __init__(self, future_id='TA888', start_datetime='2019-07-02 01:30:00', end_datetime='2021-03-05 15:00:00'):
        self.future_id = future_id
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.window = 30

        self.model = torch.load('model.pkl')
        self.Database = 'Data.db'

        self.df = self.get_data()

    def get_data(self):
        start = self.parse_datetime(self.start_datetime)
        end = self.parse_datetime(self.end_datetime)

        with sqlite3.connect(self.Database) as connection:
            cursor = connection.cursor()
            
            df = pd.read_sql_query(f"""
                                   SELECT date AS date_norm, open, high, low, close, volume
                                   FROM History
                                   WHERE date >= ?
                                   AND date <= ?
                                   AND coin == ?
                                   """, connection, params=[start, end] + [self.future_id],
                                   parse_dates=['date_norm'], index_col='date_norm')\
                                   .sort_values(['date_norm'])
        
        return df
    
    def run(self, df):
        time_list = df.index.tolist()

        hold = 0
        profit = []
        for time_idx, time in enumerate(time_list):
            if time_idx < self.window - 1:
                continue

            signal = self.apply_model(df, time_idx)

            # open long position
            if hold == 0 and signal:
                hold = 1

                open_time = time_idx

            elif hold == 0 and not signal:
                hold = -1

                open_time = time_idx

            elif hold == 1 and (not signal or time_idx - open_time == 10):
                hold = 0

                profit.append(+(df['close'].iloc[time_idx] - df['close'].iloc[open_time]))

            elif hold == -1 and (signal or time_idx - open_time == 10):
                hold = 0
                
                profit.append(-(df['close'].iloc[time_idx] - df['close'].iloc[open_time]))

        return np.array(profit)

    def apply_model(self, df, time_idx):
        data = np.array(df.iloc[time_idx - (self.window - 1):time_idx + 1]).T
        transformer = GramianAngularField()

        image = np.expand_dims(transformer.transform(data), axis=0)
        pred = F.softmax(self.model(torch.tensor(image)), dim=1).detach().numpy()
        
        signal = np.argmax(pred)
        print(signal)

        return signal
        
    @staticmethod
    def plot_profit(profit):
        y = profit.cumsum()
        plt.plot(y, color='blue')
        plt.show()

    @staticmethod
    def parse_datetime(date_time):
        return time.mktime(dt.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S').timetuple())


if __name__ == "__main__":
    futures = ['TA888', 'V888', 'RU888', 'RB888', 'C888', 'FU888', 'J888', 'L888']
    
    for future in futures:
        backtest = Backtest(future_id=future)
        
        profit = backtest.run(backtest.df)
        print(len(profit))
        backtest.plot_profit(profit)

