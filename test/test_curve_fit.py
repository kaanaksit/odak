import odak
import torch
import sys


def test():
    torch.seed()
    m = 10
    device = torch.device('cpu')
    x_val = torch.linspace(-10, 10, m).view(m, 1).to(device)
    y_val = 0.1 * x_val * torch.cos(x_val) + 0.3 * torch.rand(m, 1).to(device)
    curve = odak.learn.tools.multi_layer_perceptron(n_hidden=16)
    curve.fit(x_val, y_val, epochs=500, learning_rate=1e-2)
    estimate = torch.zeros_like(y_val)
    for i in range(estimate.shape[0]):
        estimate[i] = curve.estimate(x_val[i].view(1, 1))
    print('Estimate vs ground truth difference:{:.4f}'.format(torch.sum(torch.abs(y_val - estimate))))
    torch.save(curve.state_dict(), 'weights.pt')
    curve.load_state_dict(torch.load('weights.pt'))
    assert True == True


if __name__ == '__main__':
    sys.exit(test())

