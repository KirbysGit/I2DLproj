import matplotlib.pyplot as plt

def plot_losses(losses_cls, losses_box, save_path):
    plt.plot(losses_cls, label="Classification Loss")
    plt.plot(losses_box, label="Box Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
