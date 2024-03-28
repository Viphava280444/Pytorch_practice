print(33333)
import torch
import torch.nn as nn
print(44444)
X = torch.tensor([1,2,3,4,5], dtype=torch.float32)
Y = torch.tensor([5,10,15,20,25], dtype=torch.float32)

w1 = torch.tensor(0, dtype=torch.float32, requires_grad=True)
w0 = torch.tensor(0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w1 * x + w0

Loss = nn.MSELoss()
opt = torch.optim.SGD([w0,w1], lr=0.001)


epoch = 5


for i in range(epoch):
    y_pred = forward(X)

    loss = Loss(Y, y_pred)

    loss.backward()

    opt.step()

    opt.zero_grad

    print(f'epoch: {i + 1}, output: {y_pred}, loss: {loss}')
