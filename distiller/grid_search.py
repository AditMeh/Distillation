import json
from models.student_mnist import StudentNetMnist
from models.teacher_mnist import TeacherNetMnist

from utils import create_parser_grid_search
from train_student import distill_model
from dataloader import create_dataloaders_mnist, generate_mnist_classwise_dict
import torch


class HyperParamSearch:
    def __init__(self, config_path, teacher_network_path) -> None:
        self.hparams = load_json(config_path)
        self.teacher_network_path = teacher_network_path

    def run_single_search(self, lr, epochs, T, weight, teacher_network, device, train_dataset, val_dataset):

        student_network = StudentNetMnist().to(device=device)

        # Loading the teacher network

        train_history, val_history = distill_model(
            False, "weights/", student_network, teacher_network, lr, T, weight, epochs, train_dataset, val_dataset, device)

        return val_history[-1]

    def run_grid_search(self):
        results = {}

        device = (torch.device('cuda') if torch.cuda.is_available()
                  else torch.device('cpu'))

        print(f"Training on device {device}.")

        net = TeacherNetMnist().to(device=device)
        train_dataset = torch.load("data/MNIST/processed/training.pt")
        val_dataset = torch.load("data/MNIST/processed/test.pt")
        classwise_dict_train = generate_mnist_classwise_dict(train_dataset)
        classwise_dict_val = generate_mnist_classwise_dict(val_dataset)

        teacher_network = TeacherNetMnist()
        checkpoint = torch.load(self.teacher_network_path)
        teacher_network.load_state_dict(checkpoint)
        teacher_network = teacher_network.to(device=device)

        for key in self.hparams.keys():
            hparams = self.hparams[key]
            train_dataset, val_dataset = create_dataloaders_mnist(
                classwise_dict_train, classwise_dict_val, [i for i in range(hparams["classes"] + 1)])

            val_loss_final = self.run_single_search(
                hparams["lr"], hparams["epochs"], hparams["T"], hparams["weight"],
                teacher_network, device, train_dataset, val_dataset)

            results["lr: {}, epochs: {}, T: {}, weight: {}".format(
                hparams["lr"], hparams["epochs"], hparams["T"], hparams["weight"])] = val_loss_final

        return results


def load_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


if __name__ == "__main__":
    parser = create_parser_grid_search()
    args = parser.parse_args()

    searcher = HyperParamSearch(args.config_path, args.teacher_weights)

    results = searcher.run_grid_search()

    import pprint
    pprint.pprint(results)