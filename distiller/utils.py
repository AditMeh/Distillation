import torch


class OneHotEncoder(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, label):
        target_onehot = torch.zeros(10)
        target_onehot[label] = 1.0
        return target_onehot


class StatsTracker():
    def __init__(self):
        self.train_hist = []
        self.val_hist = []
        self.train_loss_curr = 0.0
        self.val_loss_curr = 0.0

    def update_histories(self, train_value=None, val_value=None):
        if train_value is not None:
            self.train_hist.append(train_value)
        if val_value is not None:
            self.val_hist.append(val_value)

    def update_curr_losses(self, train_value=None, val_value=None):
        if train_value is not None:
            self.train_loss_curr += train_value
        if val_value is not None:
            self.val_loss_curr += val_value

    def reset(self):
        self.train_loss_curr = 0.0
        self.val_loss_curr = 0.0
