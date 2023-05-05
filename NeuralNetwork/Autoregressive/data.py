import torch
import matplotlib.pyplot as plt

class TrainData():
    def __init__(self, batch_size=16, T=1000, num_train=600, tau=4):
        self.time = torch.arange(1, T + 1, dtype=torch.float32)
        self.x = torch.sin(0.01 * self.time) + torch.randn(T) * 0.2
        self.tau = tau
        self.num_train = num_train
        self.batch_size = batch_size
        self.T = T

    def get_dataloader(self):
        features = [self.x[i : self.T-self.tau+i] for i in range(self.tau)]
        self.features = torch.stack(features, 1)
        self.labels = self.x[self.tau:].reshape((-1, 1))
        train_data = [
            (self.features[i], self.labels[i]) for i in range(len(self.labels))
        ]
        return train_data
    
    def visualize(self):
        plt.figure(figsize=(6, 3))
        plt.plot(self.time, self.x)
        plt.xlabel('time')
        plt.ylabel('x')
        plt.xlim(1, 1000)
        plt.grid(True)
        plt.show()