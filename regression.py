import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import wandb
from simple_dataset import make_regression_dataset


class linearRegression:
    def __init__(
        self,
        data: torch.Tensor,
        lable: torch.Tensor,
        epoch: int = 10,
        optimizer: str = "adam",
        lr: float = 0.001,
        weight_decay: float = 0,
    ) -> None:
        self.data = data
        self.true_lable = lable
        self.epoch = epoch
        self.opt = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        wandb.login()
        run = wandb.init(project="Regression")

    def optimizer(self, model):
        if self.opt == "adam":
            return torch.optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

    def regression(self):
        n_sample, n_feature = self.data.shape
        output_size = 1
        model = nn.Linear(n_feature, output_size)
        return model

    def train(self):
        criterion = nn.MSELoss()
        model = self.regression()
        optimizer = self.optimizer(model)


        for i in range(self.epoch):
            y_predicted = model(self.data)
            loss = criterion(y_predicted, self.true_lable)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Epoch: {i}, Loss: {loss}")
            wandb.log({"Epoch": i, "Loss": loss})
        return model





if __name__ == "__main__":

    dataset = make_regression_dataset(100)
    x, y, x_num, y_num = dataset.generate_data()
    # dataset.plot_regression(x_num, y_num)
    regression = linearRegression(x, y,epoch=100000, lr=0.01)
    model = regression.train()
   
    predicted = model(x).detach().numpy()
    plt.plot(x_num, y_num, "ro")
    plt.plot(x_num, predicted, "b")
    plt.show()
