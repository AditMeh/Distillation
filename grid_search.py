import json
from models.student_mnist import StudentNetMnist
from models.teacher_mnist import TeacherNetMnist

from utils import create_parser_grid_search, get_classwise_performance_report
from train_student import distill_model
from dataloader import create_dataloaders_mnist, generate_mnist_classwise_dict
import torch
import os
import uuid
from copy import deepcopy


class HyperParamSearch:
    def __init__(self, config_path, teacher_network_path) -> None:
        self.hparams = load_json(config_path)
        self.teacher_network_path = teacher_network_path

    def run_single_search(self, lr, epochs, T, weight, teacher_network, device, train_dataset, val_dataset):

        student_network = StudentNetMnist().to(device=device)

        # Loading the teacher network

        train_history, val_history, best_val_loss, best_model_state = distill_model(
            False, "weights/", student_network, teacher_network, lr, T, weight, epochs, train_dataset, val_dataset, device)

        student_network.load_state_dict(best_model_state)
        return val_history[-1], best_val_loss, student_network

    def run_grid_search(self):
        results = {}
        vis_results = {"columns": [
            "lr", "T", "epochs", "weight", "best_loss"] + [str(i) for i in range(10)], "data": []}

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
                classwise_dict_train, classwise_dict_val, hparams["classes"])

            val_loss_final, best_val_loss, model = self.run_single_search(
                hparams["lr"], hparams["epochs"], hparams["T"], hparams["weight"],
                teacher_network, device, train_dataset, val_dataset)

            experiment_name = "lr: {}, epochs: {}, T: {}, weight: {}, classes: {}".format(
                hparams["lr"], hparams["epochs"], hparams["T"], hparams["weight"], hparams["classes"])

            results[experiment_name + " val loss"] = round(val_loss_final, 5)

            performance_report = get_classwise_performance_report(
                model, classwise_dict_val, device)

            performance_report = {
                x: round(performance_report[x], 5) for x in performance_report
            }

            get_vis_results(performance_report, vis_results,
                            best_val_loss, hparams)

            results[experiment_name + " classwise_performance"] = {
                x: performance_report[x] for x in performance_report.keys()}
        return results, vis_results


def get_vis_results(performance_report, vis_results, best_val_loss, hparams):
    vis_report = deepcopy(performance_report)
    vis_report["lr"] = str(hparams["lr"])
    vis_report["T"] = str(hparams["T"])
    vis_report["epochs"] = str(hparams["epochs"])
    vis_report["weight"] = str(hparams["weight"])
    vis_report["best_loss"] = str(round(best_val_loss, 5))

    vis_results["data"].append(vis_report)


def load_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


if __name__ == "__main__":
    parser = create_parser_grid_search()
    args = parser.parse_args()

    searcher = HyperParamSearch(args.config_path, args.teacher_weights)

    results, vis_results = searcher.run_grid_search()

    if not os.path.exists('json_results'):
        os.makedirs('json_results')
    if not os.path.exists('json_results/vis_results'):
        os.makedirs('json_results/vis_results')

    new_fp_json = os.path.split(args.config_path)[1].split(".")[
        0] + uuid.uuid4().hex + ".json"

    new_fp_md = os.path.split(args.config_path)[1].split(".")[
        0] + uuid.uuid4().hex + ".md"

    with open("json_results/" + new_fp_json, "w") as outfile:
        json.dump(results, outfile)
    with open("json_results/vis_results/" + new_fp_json, "w") as outfile:
        json.dump(vis_results, outfile)

    os.system(
        "python Markdown-Table-Generator/generate_markdown.py -i json_results/vis_results/{} -t json_results/vis_results/{}".format(new_fp_json, new_fp_md))
    os.remove("json_results/vis_results/" + new_fp_json)
