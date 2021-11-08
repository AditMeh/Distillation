import matplotlib.pyplot as plt


def plot_train_graph(x, num_params):

    plt.plot([i+1 for i in range(len(x))], x, label="val acc")
    plt.title("teacher val accuracy per epoch")
    plt.ylabel("val accuracy")
    plt.xlabel("epoch")
    plt.xticks(list(range(1, len(x) + 1)))
    plt.show()
