import torch
import torch.nn as nn
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
import wandb
class breast_cancer_dataset:
    def __init__(self) -> None:
        self.dataset = datasets.load_breast_cancer()
        self.X_data, self.Y_data = self.dataset.data, self.dataset.target
        self.n_samples, self.n_feature = self.X_data.shape
        wandb.login()
        wandb.init(project="BinaryClassification")

    def make_dataset(self, test_split_size:float=0.2, random_state:int=1):
        X_train, X_test, Y_trian, Y_test = train_test_split(self.X_data, self.Y_data, test_size=test_split_size, random_state=random_state)
   
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        X_train = torch.from_numpy(X_train.astype(np.float32))
        X_test = torch.from_numpy(X_test.astype(np.float32))

        Y_trian = torch.from_numpy(Y_trian.astype(np.float32))
        Y_test = torch.from_numpy(Y_test.astype(np.float32))

        Y_trian = Y_trian.view(Y_trian.shape[0], 1)
        Y_test = Y_test.view(Y_test.shape[0], 1)
        return X_train, X_test, Y_trian, Y_test

class Model(nn.Module):
    def __init__(self, n_input_features) -> None:
        super().__init__() 
        self.linear = nn.Linear(n_input_features, 1)
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
class BinaryClassification:
    def __init__(self, X_trian:torch.Tensor, X_test:torch.Tensor, Y_train:torch.Tensor, Y_test:torch.Tensor) -> None:
        self.model = Model(X_trian.shape[1])
        self.X = X_trian
        self.Y = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    
    def train(self, epoch:int=10, lr:float= 0.01):
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr= lr)

        for i in range(epoch):
            y_pred = self.model.forward(self.X)
            loss = criterion(y_pred, self.Y)
            acc = y_pred.round().eq(self.Y).sum() / float(self.Y.shape[0])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad
            print(f'Epoch: {i}, Training Loss: {loss}, Training Accuracy: {acc}')
            wandb.log({'Epoch': i, 'Training Loss': loss, 'Training Accuracy': acc})

            with torch.no_grad():
                y_pred = self.model.forward(self.X_test)
                loss = criterion(y_pred, self.Y_test)
                acc = y_pred.round().eq(self.Y_test).sum() / float(self.Y_test.shape[0])
                print(f"Epoch: {i}, Testing Loss: {loss}, Testing Accuracy: {acc}")
                wandb.log({"Testing Loss": loss, "Testing Accuracy": acc})


def main():
    cancer_data = breast_cancer_dataset()
    x_train, x_test, y_trian, y_test = cancer_data.make_dataset()
    model = BinaryClassification(x_train, x_test, y_trian, y_test)
    model.train(100)


if __name__ == '__main__':
    main()


