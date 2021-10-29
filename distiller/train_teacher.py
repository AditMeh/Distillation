from torch._C import StringType
from models.teacher_mnist import TeacherNet
from models.student_mnist import StudentNet
from dataloader import create_dataloaders_mnist
from utils import StatsTracker

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm
import os
import argparse


def train_model(save, save_dir, net, lr, epochs, train_loader, val_loader, device, batch_size=32):
    optimizer = Adam(params=net.parameters(), lr=lr)

    statsTracker = StatsTracker()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    for epoch in range(1, epochs + 1):
        statsTracker.reset()

        net.train()
        for x, labels in tqdm.tqdm(train_loader):

            x, labels = x.to(device=device), labels.to(device=device)
            outputs = net(x)
            loss = CrossEntropyLoss(reduce="mean")(outputs, labels)
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
                    reduce="mean")(val_outputs, val_labels)

                statsTracker.update_curr_losses(None, val_loss.item())

                matching = torch.eq(torch.argmax(
                    val_outputs, dim=1), val_labels)
                correct += torch.sum(matching, dim=0).item()

        train_loss_epoch = statsTracker.train_loss_curr / \
            (batch_size * len(train_loader))
        val_loss_epoch = statsTracker.val_loss_curr / \
            (batch_size * len(val_loader))
        val_accuracy = correct / (len(val_loader) * batch_size)

        statsTracker.update_histories(train_loss_epoch, None)

        statsTracker.update_histories(None, val_loss_epoch)

        print('Teacher_network_Epoch {}, Train Loss {}, Val Loss {}, Val Accuracy {}'.format(
            epoch, train_loss_epoch, val_loss_epoch, val_accuracy))

        scheduler.step(val_loss_epoch)

    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(net.state_dict(), os.path.join(
            save_dir, 'Teacher_network_val_loss{}_val_accuracy{}'.format(round(val_loss_epoch, 3), round(val_accuracy, 3))))

    return statsTracker


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--save", action="store_true", default=False)

    parser.add_argument(
        "save_dir", help="directory to save the model", type=str)

    parser.add_argument("lr", help="learning rate", type=float)

    parser.add_argument("epochs", help="epochs", type=int)

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    print(f"Training on device {device}.")

    args = parser.parse_args()

    train_dataset, val_dataset = create_dataloaders_mnist()
    train_model(args.save, args.save_dir, TeacherNet().to(
        device=device), args.lr, args.epochs, train_dataset, val_dataset, device)
