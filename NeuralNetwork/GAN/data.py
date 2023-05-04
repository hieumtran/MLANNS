import torch

import math
import matplotlib.pyplot as plt

torch.manual_seed(111)

def get_traindata(data_length):
    train_data = torch.zeros((data_length, 2))
    train_data[:, 0] = 2 * math.pi * torch.rand(data_length)
    train_data[:, 1] = torch.sin(train_data[:, 0])
    train_labels = torch.zeros(data_length)
    train_set = [
        (train_data[i], train_labels[i]) for i in range(data_length)
    ]
    return train_data, train_set

def viz(train_data):
    plt.plot(train_data[:, 0], train_data[:, 1], ".")
    plt.show()

