import matplotlib.pyplot as plt


def plot_train_graph(train_history, val_history, num_params):
    plt.title(label="Model with {} params".format(num_params))
    plt.plot([i+1 for i in range(len(train_history))],
             train_history, label="train loss")
    plt.plot([i+1 for i in range(len(val_history))],
             val_history, label="val loss")
    plt.title("teacher val accuracy per epoch")
    plt.ylabel("val accuracy")
    plt.xlabel("epoch")
    plt.xticks([1] + list(range(5, len(train_history) + 1, 5)))
    plt.legend()
    plt.show()


def display_image_grid(images):
    fig, axs = plt.subplots(1, 10, figsize=(8, 8))

    for i, img in enumerate(images):
        axs[i].imshow(images[i])

    plt.tight_layout()
    plt.show()
