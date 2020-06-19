import pandas as pd
import numpy as np

from rcnn.helper import reduce_mem_usage, LabelEncoderExt


def label_encoding(cfg, df, le=None):
    le = {} if le is None else le
    for col in cfg['cat_static'] + cfg['cat_seq']:
        if col not in le.keys():
            le[col] = LabelEncoderExt()
            le[col].fit(df[col])
            n_class = len(le[col].classes_)
            cfg['n_class'].append(n_class)
            cfg['embd_size'].append(min(50, n_class // 2 + 1))
        df[col] = le[col].transform(df[col])
    return df, le


def preprocess(cfg, df, le=None):
    for col in cfg['cat_static'] + cfg['cat_seq']:
        df[col] = df[col].fillna(np.nan).astype(str)
    df, le = label_encoding(cfg, df, le)

    df[cfg['con']] = df[cfg['con']].fillna(0)
    return reduce_mem_usage(df), le


def merge_data(df_sales, calendar, sell_prices, drop_leading_zeros=False):
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    df_sales = pd.melt(df_sales, id_vars=id_vars, var_name='d', value_name='demand')

    df_sales = pd.merge(df_sales, calendar, on='d')
    df_sales = df_sales.sort_values(['id', 'date']).reset_index(drop=True)
    df_sales['idx'] = df_sales.index
    df_sales['not_zero'] = df_sales['demand'].ne(0)

    if drop_leading_zeros:
        starts = df_sales.groupby('id')['not_zero'].idxmax()
        ends = df_sales.groupby('id')['idx'].max()
        list_idx = []
        for start, end in zip(starts.values, ends.values):
            idx = range(start, end + 1)
            list_idx += idx
        df_sales = df_sales.loc[list_idx]
    df_sales = pd.merge(df_sales, sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'])
    return df_sales.reset_index(drop=True).drop('idx', axis=1)


def prepare_trainset(cfg, df_sales, calendar, sell_prices):
    df = merge_data(df_sales, calendar, sell_prices, drop_leading_zeros=True)
    df, le = preprocess(cfg, df)
    df_valid = df[df['date'] >= cfg['val_start']]
    df_train = df[df['date'] < cfg['val_start']]
    return df_train, df_valid, le


def prepare_testset(cfg, df_sales, calendar, sell_prices, le):
    max_d = df_sales.columns[-1]
    max_d = int(max_d.split('_')[1])
    for i in range(cfg['out_steps']):
        pred_d = 'd_' + str(i + max_d + 1)
        df_sales[pred_d] = 0
    seq_len = cfg['in_steps'] + cfg['out_steps']
    cols = ['id'] + cfg['cat_static'] + df_sales.columns[-seq_len:].to_list()
    df_sales = df_sales[cols]
    df = merge_data(df_sales, calendar, sell_prices)
    df, _ = preprocess(cfg, df, le)
    return df
