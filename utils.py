import argparse
import torch
import copy
import torch.nn.functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_classwise_performance_report(model, classwise_dict, device):

    performance_dict = {}

    for c in classwise_dict:
        curr_dataset = torch.stack(classwise_dict[c]).type(
            torch.float32).to(device=device)

        curr_labels = (torch.ones(size=(len(curr_dataset), ))
                       * c).to(device=device)

        with torch.no_grad():
            model.eval()
            val_student_logits = model(curr_dataset)

            val_softmax_student = F.softmax(val_student_logits, dim=1)
            matching = torch.eq(torch.argmax(
                val_softmax_student, dim=1), curr_labels)
            performance_dict[c] = torch.sum(
                matching, dim=0).item()/len(matching)
    return performance_dict


def create_parser_train_student():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--save", action="store_true", default=False)

    parser.add_argument(
        "teacher_weights", help="filepath to the weights of the teacher model", type=str)
    parser.add_argument(
        "save_dir", help="directory to save the model", type=str)

    parser.add_argument("lr", help="learning rate", type=float)

    parser.add_argument("T", help="softmax temperature", type=float)

    parser.add_argument("classes", default=9,
                        help="classes to train the network on", type=int)

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


def create_parser_grid_search():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_path", help="path to grid search file", type=str)
    parser.add_argument(
        "teacher_weights", help="weights for teacher network", type=str)
    return parser
