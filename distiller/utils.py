import argparse
import torch
import copy


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


class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best_val_loss = torch.inf
        self.stop = False
        self.epoch_counts = 0
        self.best_model = None

    def __call__(self, val_loss, net):

        if self.best_val_loss > val_loss + self.delta:
            self.best_val_loss = val_loss
            self.epoch_counts = 0
            self.store_model(net)

        else:
            self.epoch_counts += 1

            if self.epoch_counts == self.patience:
                self.stop = True

    def store_model(self, net):
        self.best_model = copy.deepcopy(net.state_dict())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_parser_train_student():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--save", action="store_true", default=False)

    parser.add_argument(
        "teacher_weights", help="filepath to the weights of the teacher model", type=str)
    parser.add_argument(
        "save_dir", help="directory to save the model", type=str)

    parser.add_argument("lr", help="learning rate", type=float)

    parser.add_argument("T", help="softmax temperature", type=float)

    parser.add_argument(
        "weight", help="weight given to soft target loss term in the distillation loss", type=float)

    parser.add_argument("epochs", help="epochs", type=int)

    return parser


def create_parser_train_teacher():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--save", action="store_true", default=False)

    parser.add_argument(
        "save_dir", help="directory to save the model", type=str)

    parser.add_argument("lr", help="learning rate", type=float)

    parser.add_argument("epochs", help="epochs", type=int)
    return parser
