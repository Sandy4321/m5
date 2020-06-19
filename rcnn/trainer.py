import copy
import time
import torch
import numpy as np
from rcnn.model import RCNN, device


class EarlyStopping:
    def __init__(self, patience, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
        """
        self.patience = patience
        self.delta = delta
        self.early_stop = False
        self.counter = 0
        self.epoch = 0
        self.best_epoch = 0
        self.best_score = None
        self.best_model = None
        self.best_train_loss = np.Inf
        self.best_val_loss = np.Inf

    def __call__(self, train_loss, val_loss, model):
        self.epoch += 1
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(train_loss, val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Early stopping counter: {self.counter} out of {self.patience} patience')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(train_loss, val_loss, model)
            self.counter = 0

    def save_checkpoint(self, train_loss, val_loss, model):
        self.best_model = copy.deepcopy(model)
        self.best_train_loss = train_loss
        self.best_val_loss = val_loss
        self.best_epoch = self.epoch


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = RCNN(cfg)
        self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        self.loss_function = torch.nn.L1Loss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg['lr'])
        self.early_stopping = EarlyStopping(cfg['patience'])
        self.train_losses = []
        self.val_losses = []

    def fit(self, train_loader, valid_loader, writer=None):
        if writer:
            writer.watch(self.model, log='all')

        for i in range(self.cfg['epoch']):
            total_train_loss = 0
            self.model.train()
            start_time = time.time()
            for batch in train_loader:
                x = batch[0]
                x_seq = batch[1]
                y = batch[2].float().to(device)
                self.optimizer.zero_grad()
                output = self.model(x, x_seq)
                loss = self.loss_function(output, y)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
            train_loss = total_train_loss / len(train_loader)
            self.train_losses.append(train_loss)

            self.model.eval()
            with torch.no_grad():
                total_val_loss = 0
                for batch in valid_loader:
                    x = batch[0]
                    x_seq = batch[1]
                    y = batch[2].float().to(device)
                    output = self.model(x, x_seq)
                    loss = self.loss_function(output, y)
                    total_val_loss += loss.item()
                val_loss = total_val_loss / len(valid_loader)
                self.val_losses.append(val_loss)

            time_spent = time.time() - start_time
            print('epoch %d/%d training loss: %f / validation loss: %f (%d sec)' % (
                i + 1, self.cfg['epoch'], train_loss, val_loss, time_spent))
            if writer:
                writer.log({f"train_loss": self.early_stopping.best_train_loss,
                            f"val_loss": self.early_stopping.best_val_loss,
                            f"best_epoch": self.early_stopping.best_epoch})

            self.early_stopping(train_loss, val_loss, self.model)
            if self.early_stopping.early_stop:
                break

        if writer:
            writer.unwatch(self.model)
        self.model.load_state_dict(self.early_stopping.best_model.state_dict())
        print(f'Loading best model checkpoint at epoch {self.early_stopping.best_epoch}')
        return self.model

