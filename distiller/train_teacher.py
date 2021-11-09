from torch.nn.modules.loss import CrossEntropyLoss
from models.teacher_mnist import TeacherNetMnist
from dataloader import create_dataloaders_mnist
from utils import StatsTracker, count_parameters, create_parser_train_teacher, EarlyStopping

import torch
from torch.nn.functional import softmax
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm
import os

from visualization.plot_train_graph import plot_train_graph


def train_model(save, save_dir, net, lr, epochs, train_loader, val_loader, device, batch_size=32):
    optimizer = Adam(params=net.parameters(), lr=lr)

    statsTracker = StatsTracker()
    earlyStopping = EarlyStopping(patience=8, delta=0.0)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, eps=0.0003, verbose=True)

    for epoch in range(1, epochs + 1):
        statsTracker.reset()

        net.train()
        for x, labels in tqdm.tqdm(train_loader):

            x, labels = x.to(device=device), labels.to(device=device)
            outputs = net(x)
            loss = CrossEntropyLoss(reduction="mean")(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            statsTracker.update_curr_losses(loss.item(), None)

        correct = 0

        with torch.no_grad():
            net.eval()
            for val_x, val_labels in tqdm.tqdm(val_loader):
                val_x, val_labels = val_x.to(
                    device=device), val_labels.to(device=device)
                val_outputs = net(val_x)
                val_loss = CrossEntropyLoss(
                    reduction="mean")(val_outputs, val_labels)

                statsTracker.update_curr_losses(None, val_loss.item())

                matching = torch.eq(torch.argmax(
                    softmax(val_outputs, dim=1), dim=1), val_labels)
                correct += torch.sum(matching, dim=0).item()

        train_loss_epoch = statsTracker.train_loss_curr / \
            (batch_size * len(train_loader))
        val_loss_epoch = statsTracker.val_loss_curr / \
            (batch_size * len(val_loader))
        val_accuracy = correct / (len(val_loader) * batch_size)

        statsTracker.update_histories(train_loss_epoch, None)

        statsTracker.update_histories(None, val_loss_epoch)

        print('Teacher_network: Epoch {}, Train Loss {}, Val Loss {}, Val Accuracy {}'.format(
            epoch, round(train_loss_epoch, 5), round(val_loss_epoch, 5), round(val_accuracy, 5)))

        scheduler.step(val_loss_epoch)

        earlyStopping(val_loss_epoch, net)

        if earlyStopping.stop:
            break

    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(earlyStopping.best_model, os.path.join(
            save_dir, 'Teacher_network_val_loss{}'.format(round(val_loss_epoch, 5))))

    return statsTracker.train_hist, statsTracker.val_hist


if __name__ == "__main__":

    parser = create_parser_train_teacher()
    args = parser.parse_args()

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    print(f"Training on device {device}.")

    train_dataset, val_dataset = create_dataloaders_mnist()
    net = TeacherNetMnist().to(device=device)
    
    train_hist, val_hist = train_model(args.save, args.save_dir, net, args.lr,
                                       args.epochs, train_dataset, val_dataset, device)
    
    plot_train_graph(train_hist, val_hist, count_parameters(net))
