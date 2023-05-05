from data import *
import torch.nn as nn
from model import *
import matplotlib.pyplot as plt
import numpy as np

def train(
    train_data, 
    lr,
    batch_size, 
    shuffle,
    num_epoch,
    model
):
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=shuffle
    )

    # loss function
    loss = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        for batch_input, batch_output in train_loader:
            model.zero_grad()
            model_output = model(batch_input)
            local_loss = loss(model_output, batch_output)
            local_loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}: {local_loss}')

def convert_numpy(item):
    return item.detach().numpy()

def evaluate_one_step(input, output, model):
    predict = model(input)

    predict = convert_numpy(predict)
    output = convert_numpy(output)

    plt.figure(figsize=(6,3))
    plt.plot(output, label='True')
    plt.plot(predict, label='1-step', color='red')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = TrainData()
    # data.visualize()
    train_data = data.get_dataloader()

    # Hyperparameters
    batch_size = 32
    shuffle = False
    num_epoch = 30
    lr = 0.001

    # Init model
    linear_model = LinearRegression(4, 1)
    train(
        train_data,
        lr=lr,
        batch_size=batch_size,
        shuffle=shuffle,
        num_epoch=num_epoch,
        model=linear_model
    )

    evaluate_one_step(data.features, data.labels, linear_model)
