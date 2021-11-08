from models.teacher_mnist import TeacherNetMnist
from models.student_mnist import StudentNetMnist
from dataloader import create_dataloaders_mnist, create_onehot_dataloaders_mnist
from distiller import distillation_loss
from utils import StatsTracker, create_parser_train_student, count_parameters, EarlyStopping
from visualization import plot_train_graph

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm
import os
import argparse


def distill_model(save, save_dir, student_net, teacher_net, lr, T, weight, epochs, train_loader, val_loader, device, batch_size=32):
    optimizer = Adam(params=student_net.parameters(), lr=lr)

    statsTracker = StatsTracker()
    earlyStopping = EarlyStopping(patience=4, delta=0.0)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    for epoch in range(1, epochs + 1):
        statsTracker.reset()

        student_net.train()
        for x, labels in tqdm.tqdm(train_loader):
            x, labels = x.to(device=device), labels.to(device=device)

            student_logits = student_net(x)

            with torch.no_grad():
                teacher_logits = teacher_net(x)
                teacher_logits_T = teacher_logits/T
                softmax_teacher = F.softmax(teacher_logits_T, dim=1)

                student_ce_loss = CrossEntropyLoss(
                    reduction="mean")(student_logits, labels)
            student_logits_T = student_logits/T
            DL_loss = distillation_loss(
                student_logits, softmax_teacher, labels, weight)
            optimizer.zero_grad()
            DL_loss.backward()
            optimizer.step()
            statsTracker.update_curr_losses(student_ce_loss.item(), None)

        correct = 0

        with torch.no_grad():
            student_net.eval()
            for val_x, val_labels in tqdm.tqdm(val_loader):
                val_x, val_labels = val_x.to(
                    device=device), val_labels.to(device=device)
                val_student_logits = student_net(val_x)

                val_softmax_student = F.softmax(val_student_logits, dim=1)

                val_loss = CrossEntropyLoss(reduction="mean")(
                    val_softmax_student, val_labels)

                statsTracker.update_curr_losses(None, val_loss.item())

                matching = torch.eq(torch.argmax(
                    val_softmax_student, dim=1), torch.argmax(val_labels, dim=1))
                correct += torch.sum(matching, dim=0).item()

        train_loss_epoch = statsTracker.train_loss_curr / \
            (batch_size * len(train_loader))
        val_loss_epoch = statsTracker.val_loss_curr / \
            (batch_size * len(val_loader))
        val_accuracy = correct / (len(val_loader) * batch_size)

        statsTracker.update_histories(train_loss_epoch, None)

        statsTracker.update_histories(None, val_loss_epoch)

        print('Student_network, Epoch {}, Train Loss {}, Val Loss {}, Val Accuracy {}'.format(
            epoch, round(train_loss_epoch, 6), round(val_loss_epoch, 6), round(val_accuracy, 6)))

        scheduler.step(val_loss_epoch)
        earlyStopping(val_loss_epoch, student_net)

        if earlyStopping.stop:
            break
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(student_net.state_dict(), os.path.join(
            save_dir, 'Student_network_val_loss{}_val_accuracy{}'.format(round(val_loss_epoch, 3), round(val_accuracy, 3))))

    return statsTracker.train_hist, statsTracker.val_hist


if __name__ == "__main__":

    parser = create_parser_train_student()
    args = parser.parse_args()

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    print(f"Training on device {device}.")

    train_dataset, val_dataset = create_onehot_dataloaders_mnist()
    student_network = StudentNetMnist().to(device=device)

    # Loading the teacher network
    teacher_network = TeacherNetMnist()
    checkpoint = torch.load(args.teacher_weights)
    teacher_network.load_state_dict(torch.load(args.teacher_weights))
    teacher_network = teacher_network.to(device=device)

    print('Student Model: {} params, Teacher Model: {} params'.format(
        count_parameters(student_network), count_parameters(teacher_network)))

    train_history, val_history = distill_model(args.save, args.save_dir, student_network, teacher_network,
                                               args.lr, args.T, args.weight, args.epochs, train_dataset, val_dataset, device)

    plot_train_graph(val_history, count_parameters(student_network))
