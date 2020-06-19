import random
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def reduce_mem_usage(df):
    start_time = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory decreased to {:5.2f} Mb ({:.1f}% reduction), time spent {:.1f}s'.format(
        end_mem, 100 * (start_mem - end_mem) / start_mem, time.time() - start_time))
    return df


def read_data(base_dir):
    calendar = pd.read_csv(base_dir + 'calendar.csv')
    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))
    calendar = reduce_mem_usage(calendar)

    sell_prices = pd.read_csv(base_dir + 'sell_prices.csv')
    print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))
    sell_prices = reduce_mem_usage(sell_prices)

    sales_train_validation = pd.read_csv(base_dir + 'sales_train_validation.csv')
    print('Sales train validation has {} rows and {} columns'.format(
        sales_train_validation.shape[0], sales_train_validation.shape[1]))

    sales_train_evaluation = pd.read_csv(base_dir + 'sales_train_evaluation.csv')
    print('Sales train evaluation has {} rows and {} columns'.format(
        sales_train_evaluation.shape[0], sales_train_evaluation.shape[1]))

    submission = pd.read_csv(base_dir + '/sample_submission.csv')
    return calendar, sell_prices, sales_train_validation, sales_train_evaluation, submission


class LabelEncoderExt(object):
    def __init__(self, unknown_label='Unknown'):
        self.label_encoder = LabelEncoder()
        self.unknown_label = unknown_label
        self.classes_ = None

    def fit(self, data_list):
        self.label_encoder = self.label_encoder.fit(list(data_list) + [self.unknown_label])
        self.classes_ = self.label_encoder.classes_
        return self

    def transform(self, data_list):
        new_data_list = list(data_list)
        unknown_list = [item for item in np.unique(data_list) if item not in self.label_encoder.classes_]
        if len(unknown_list) > 0:
            print("Unknown %s: %s" % (data_list.name, [str(x) for x in unknown_list]))
            new_data_list = [self.unknown_label if x in unknown_list else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)

    def inverse_transform(self, data_list):
        return self.label_encoder.inverse_transform(list(data_list))
