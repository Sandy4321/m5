import time
import yaml
from datetime import datetime
from functools import partial

import wandb
from torch.utils.data import DataLoader

from rcnn.dataset import SequenceGenerator, pad_collate
from rcnn.helper import read_data, set_seed
from rcnn.preprocess import prepare_trainset, prepare_testset
from rcnn.trainer import Trainer

DEBUG = 1


def setup():
    with open('config.yml') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def prepare_data(cfg):
    calendar, sell_prices, _, sales_train_evaluation, submission = read_data(cfg['data_dir'])
    if DEBUG:
        cfg['epoch'] = 2
        debug_ids = ['HOBBIES_1_001_CA_1_evaluation', 'HOUSEHOLD_1_001_CA_1_evaluation']
        sales_train_evaluation = sales_train_evaluation[sales_train_evaluation['id'].isin(debug_ids)]

    df_train, df_valid, le = prepare_trainset(cfg, sales_train_evaluation, calendar, sell_prices)
    df_eval = prepare_testset(cfg, sales_train_evaluation, calendar, sell_prices, le)

    train_seq = SequenceGenerator(cfg, df_train)
    valid_seq = SequenceGenerator(cfg, df_valid)
    eval_seq = SequenceGenerator(cfg, df_eval)

    set_seed(cfg['seed'])
    collate_fn = partial(pad_collate, cfg=cfg)
    train_loader = DataLoader(train_seq, cfg['bs'], shuffle=True, collate_fn=collate_fn, drop_last=True)
    valid_loader = DataLoader(valid_seq, cfg['bs'], shuffle=False, collate_fn=collate_fn, drop_last=False)
    eval_loader = DataLoader(eval_seq, cfg['bs'], shuffle=False, collate_fn=collate_fn, drop_last=False)
    return train_loader, valid_loader, eval_loader, le


def train(cfg, train_loader, valid_loader):
    writer = None
    if not DEBUG:
        exp_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        writer = wandb.init(dir=cfg['data_dir'], config=cfg, project='m5', id=exp_id, save_code=True)
    print('Start training')
    start_time = time.time()
    set_seed(cfg['seed'])
    trainer = Trainer(cfg)
    model = trainer.fit(train_loader, valid_loader, writer)
    time_spent = time.time() - start_time
    print('Finished training, total time spent: %.2f hrs' % (time_spent / 3600))
    return model


def evaluate(cfg, eval_loader, submission, le):
    return


if __name__ == '__main__':
    cfg = setup()
    train_loader, valid_loader, eval_loader, le = prepare_data(cfg)
    model = train(cfg, train_loader, valid_loader)
