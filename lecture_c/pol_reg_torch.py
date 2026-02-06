import torch
import torch.nn as nn
from torch import optim

import matplotlib.pyplot as plt
import numpy as np


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 1, bias = False)
    
    def forward(self, x):
        return self.linear1(x)



def poly(x, W):
    k = W.shape[-1]
    X = create_desingmatrix(x.T, k-1)
    return W@X.T

def create_desingmatrix(x_train, k):
    X = torch.zeros((x_train.shape[0], k+1))
    for i in range(k+1):
        X[:, i] = x_train**i
    return X

def closed_form(x_train, y_train, k = 5):
    X = create_desingmatrix(x_train, k)
    Y = y_train

    W = Y.T@X@torch.linalg.inv(X.T@X)
    return W





if __name__ == "__main__":
    #1 a)

    # DATA PREP 
    N_TRAIN = 15
    SIGMA_NOISE = 0.1

    torch.manual_seed(0xDEADBEEF)
    x_train = torch.rand(N_TRAIN) * 2 * torch.pi
    y_train = torch.sin(x_train) + torch.randn(N_TRAIN) * SIGMA_NOISE

    X_train = torch.vstack([x_train**i for i in range(4)]).T

    # study lrs
    learning_rates = [10**i for i in range(-10, -4)]

    for lr in learning_rates:
        model  = LinearModel()
        with torch.no_grad():
            model.linear1.weight.fill_(1)

        loss_function = nn.MSELoss()
        sgd = optim.SGD(model.parameters(), lr=lr)

        loss_history = []

        for i in range(100):
            preds   = model(X_train)

            loss = loss_function(preds.flatten(), y_train)
            loss.backward()

            sgd.step()
            sgd.zero_grad()

            loss_history.append(loss.detach())
        plt.plot(loss_history, label = f"lr = {lr}")
    
    plt.legend()
    plt.ylabel("MSE Loss")
    plt.xlabel("#steps")
    plt.title("Loss curves for different learning rates")
    plt.savefig("plots/Loss curves for different learning rates")
    plt.show()

    x = torch.linspace(0, 2 * torch.pi, 100, requires_grad=True)
    X = torch.vstack([x**i for i in range(4)]).T

    y = model(X)

    p = closed_form(x_train, y_train, 3)
    fitted_poly_y = poly(x, p)

    x_plot = x.detach()

    plt.plot(x_plot, fitted_poly_y.detach(), label = "Least square solution", c = "cyan")
    plt.plot(x_plot, y.detach(),             label = f"SGD with lr = {lr}"  , c = "red")
    plt.plot(x_plot, np.sin(x.detach()),     label = "True sine"            , c = "black")
    plt.xlabel("x")
    plt.xlabel("y")

    plt.scatter(x_train.detach(), y_train.detach(), label = "Training data")

    plt.title("Model trained with SGD vs Least square solution")
    plt.savefig("plots/Model trained with SGD vs Least square solution")
    plt.legend()
    plt.show()

    #1 b)

    hess = 2 * (X_train.T@X_train)
    eigen_values = torch.linalg.eigvals(hess).real
    condition_number = eigen_values.max()/eigen_values.min()

    print("Conditionumber is:", condition_number)

    model  = LinearModel()

    lr = 7*(10**-5)

    #Weight initialization 1, 0.1, 0.01, 0.001
    with torch.no_grad():
        w = model.linear1.weight
        numel = w.numel()
        powers = torch.arange(numel, dtype=w.dtype, device=w.device)
        w.copy_((0.1 ** powers).view_as(w))


    loss_function = nn.MSELoss()
    sgd = optim.SGD(model.parameters(), lr=lr, momentum = 0.90)

    loss_history = []

    for i in range(100):
        preds   = model(X_train)
        targets = y_train

        loss = loss_function(preds.flatten(), targets)
        loss.backward()

        sgd.step()
        sgd.zero_grad()

        loss_history.append(loss.detach())

    plt.plot(loss_history, label = f"lr = {lr}")
    plt.legend()
    plt.title("Model trained with SGD and momentum")
    plt.ylabel("MSE Loss")
    plt.xlabel("#steps")
    plt.savefig("plots/Model trained with SGD and momentum")
    plt.show()

    x = torch.linspace(0, 2 * torch.pi, 100, requires_grad=True)
    X = torch.vstack([x**i for i in range(4)]).T

    y = model(X)

    fitted_poly_y = poly(x, p)

    x_plot = x.detach()

    plt.plot(x_plot, fitted_poly_y.detach(), label = "Least square solution", c = "cyan")
    plt.plot(x_plot, y.detach(),             label = "Model aprox", c = "red")
    plt.plot(x_plot, np.sin(x.detach()),     label = "True sine", c = "black")

    plt.scatter(x_train.detach(), y_train.detach(), label = "Training data")
    plt.title("Model trained with SGD and momentum vs Least square solution")
    plt.legend()
    plt.savefig("plots/Model trained with SGD and momentum vs Least square solution")
    plt.show()

    model  = LinearModel()

    lr = 0.2

    #Weight initialization 1, 0.1, 0.01, 0.001
    with torch.no_grad():
        w = model.linear1.weight
        numel = w.numel()
        powers = torch.arange(numel, dtype=w.dtype, device=w.device)
        w.copy_((1 ** powers).view_as(w))


    loss_function = nn.MSELoss()
    sgd = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))

    loss_history = []

    for i in range(100):
        preds   = model(X_train)
        targets = y_train

        loss = loss_function(preds.flatten(), targets)
        loss.backward()

        sgd.step()
        sgd.zero_grad()

        loss_history.append(loss.detach())

    plt.plot(loss_history, label = f"lr = {lr}")
    plt.legend()
    plt.ylabel("MSE Loss")
    plt.xlabel("#steps")
    plt.title("Loss curve for model trained with ADAM")
    plt.savefig("plots/Loss curve for model trained with ADAM")
    plt.show()

    x = torch.linspace(0, 2 * torch.pi, 100, requires_grad=True)
    X = torch.vstack([x**i for i in range(4)]).T

    y = model(X)

    fitted_poly_y = poly(x, p)

    x_plot = x.detach()

    plt.plot(x_plot, fitted_poly_y.detach(), label = "Least square solution", c = "cyan")
    plt.plot(x_plot, y.detach(),             label = "Model aprox", c = "red")
    plt.plot(x_plot, np.sin(x.detach()),     label = "True sine", c = "black")

    plt.scatter(x_train.detach(), y_train.detach(), label = "Training data")
    plt.title("Model trained with ADAM vs Least square solution")
    plt.savefig("plots/Model trained with ADAM vs Least square solution")
    plt.legend()
    plt.show()

    model = LinearModel()

    # sensible init
    with torch.no_grad():
        model.linear1.weight.fill_(0.0)

    loss_function = nn.MSELoss()

    optimizer = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=100,
        history_size=10,
        line_search_fn="strong_wolfe"
    )

    loss_history = []

    def closure():
        optimizer.zero_grad()
        preds = model(X_train)
        loss = loss_function(preds.flatten(), y_train)
        loss.backward()
        loss_history.append(loss.detach())
        return loss

    optimizer.step(closure)

    final_loss = loss_history[-1].item()
    print("Final LBFGS loss:", final_loss)
    y_lbfgs = model(X)

    p = closed_form(x_train, y_train, 3)
    fitted_poly_y = poly(x, p)

    plt.plot(x_plot, fitted_poly_y.detach(), label="Least squares solution", c="cyan")
    plt.plot(x_plot, y_lbfgs.detach(), label="LBFGS", c="red")
    plt.plot(x_plot, np.sin(x_plot), label="True sine", c="black")
    plt.scatter(x_train, y_train, label="Training data")
    plt.legend()
    plt.title("LBFGS vs Least Squares")
    plt.savefig("plots/LBFGS vs Least Squares")
    plt.show()



