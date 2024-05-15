import matplotlib.pyplot as plt


def plot_experiment(data, xlabel, ylabel, title):
    plt.figure(figsize=(10, 5))
    plt.plot(data, label=ylabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

