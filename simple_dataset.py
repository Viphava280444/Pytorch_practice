import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import wandb


class make_regression_dataset:
    def __init__(
        self,
        num_data: int = 10,
        num_feature: int = 1,
        noise: int = 10,
        random_state: int = 1,
    ) -> None:
        self.num_data = num_data
        self.num_feature = num_feature
        self.noise = noise
        self.random_state = random_state

    def generate_data(self):
        X_numpy, Y_numpy = datasets.make_regression(
            self.num_data,
            self.num_feature,
            noise=self.noise,
            random_state=self.random_state,
        )
        X = torch.from_numpy(X_numpy.astype(np.float32))
        Y = torch.from_numpy(Y_numpy.astype(np.float32))
        Y = Y.view(Y.shape[0], 1)
        return X, Y, X_numpy, Y_numpy

    def plot_regression(self, x, y, option="o"):
        plt.plot(x, y, option)
        plt.show()