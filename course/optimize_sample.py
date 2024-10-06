import torch
import odak
import sys


def forward(x, m, n):
    y = m * x + n
    return y


def main():
    m = torch.tensor([100.], requires_grad = True)
    n = torch.tensor([0.], requires_grad = True)
    x_vals = torch.tensor([1., 2., 3., 100.])
    y_vals = torch.tensor([5., 6., 7., 101.])
    optimizer = torch.optim.Adam([m, n], lr = 5e1)
    loss_function = torch.nn.MSELoss()
    for step in range(1000):
        optimizer.zero_grad()
        y_estimate = forward(x_vals, m, n)
        loss = loss_function(y_estimate, y_vals)
        loss.backward(retain_graph = True)
        optimizer.step()
        print('Step: {}, Loss: {}'.format(step, loss.item()))
    print(m, n)


if __name__ == '__main__':
    sys.exit(main())
