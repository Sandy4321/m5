import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceGenerator(Dataset):

    def __init__(self, cfg, df):
        self.cfg = cfg
        self.seq_len = cfg['in_steps'] + cfg['out_steps']
        self.df = self.__get_idx(df)
        self.cols = cfg['cat_static'] + cfg['cat_seq'] + cfg['con']
        self.data = self.df[self.cols].values

    def __get_idx(self, df):
        df = df.sort_values(['id', 'date']).reset_index(drop=True)
        df['idx'] = df.index
        df_group_idx = df.groupby(['id'])['idx']
        self.grp_min_idx = df_group_idx.min().values
        self.grp_max_idx = df_group_idx.max().values
        self.idx_start = []
        self.idx_end = []

        for i, j in zip(self.grp_min_idx, self.grp_max_idx):  # start and end row index for each time-series
            j = j + 1
            row_count = j - i
            if row_count >= self.seq_len:
                self.idx_start += list(range(i, j - self.seq_len + 1))
                self.idx_end += list(range(i + self.seq_len, j + 1))
        assert len(self.idx_start) == len(self.idx_end)
        return df

    def __len__(self):
        return len(self.idx_start)

    def __getitem__(self, index):
        start = self.idx_start[index]
        end = self.idx_end[index]
        static = self.data[start, :len(self.cfg['cat_static'])]
        seq = self.data[start:end, len(self.cfg['cat_static']):]
        return static, seq


def pad_collate(batch, cfg):
    data = [list(samples) for samples in zip(*batch)]
    x = torch.from_numpy(np.array(data[0]))
    x_seq = torch.from_numpy(np.array(data[1]))
    y = x_seq[:, -cfg['out_steps']:, -1:].clone().detach()
    x_seq[:, -cfg['out_steps']:, -1:] = 0
    return x, x_seq, y
