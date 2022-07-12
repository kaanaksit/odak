# odak.learn.tools.multi_layer_perceptron

::: odak.learn.tools.multi_layer_perceptron
    selection:
        docstring_style: numpy

## Notes

Regarding usage of this definition, you can get inspiration from below script:
``` python
import sys
import odak
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


def test():
    torch.seed()
    m = 1000
    device = torch.device('cuda')
    x_val = torch.linspace(-10, 10, m).view(m, 1).to(device)
    y_val = 0.1 * x_val * torch.cos(x_val) + 0.3 * torch.rand(m, 1).to(device)
    curve = odak.learn.tools.multi_layer_perceptron(n_hidden=512, device=device)
    curve.fit(x_val, y_val, epochs=10000, learning_rate=1e-4)
    estimate = torch.zeros_like(y_val)
    for i in range(estimate.shape[0]):
        estimate[i] = curve.forward(x_val[i].view(1, 1))
    fig, ax = plt.subplots(1)
    x_val = x_val.detach().cpu().view(m).numpy()
    y_val = y_val.detach().cpu().view(m).numpy()
    estimate = estimate.detach().cpu().view(m).numpy()
    ax.plot(x_val, y_val, color='r')
    ax.plot(x_val, estimate, color='b')
    plt.show()


if __name__ == '__main__':
    sys.exit(test())
```


## See also

* [`General Toolkit`](../../../toolkit.md)

