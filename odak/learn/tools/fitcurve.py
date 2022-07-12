import torch
import torch.nn.functional as F
from tqdm import tqdm


class multi_layer_perceptron():
    """
    A flexible class to correlate one dimensional input data to an one dimensional output data.
    """
    def __init__(self, n_input=1, n_hidden=64, n_output=1, device=None):
        """
        Parameters
        ----------
        n_input         : int
                          Input size [a].
        n_hidden        : int
                          Hidden layer size [b].
        n_output        : int
                          Output size [c].
        """
        super(multi_layer_perceptron, self).__init__()
        self.device = device
        if isinstance(self.device, type(None)):
            self.device = torch.device('cpu')
        self.loss_function = torch.nn.MSELoss()
        self.layer0 = torch.nn.Linear( n_input, n_hidden).to(self.device)
        self.layer1 = torch.nn.Linear(n_hidden, n_hidden).to(self.device)
        self.layer2 = torch.nn.Linear(n_hidden, n_hidden).to(self.device)
        self.layer3 = torch.nn.Linear(n_hidden, n_hidden).to(self.device)
        self.layer4 = torch.nn.Linear(n_hidden, n_output).to(self.device)
        self.activation = torch.nn.ReLU()
        self.parameters = [
                           self.layer0.weight,
                           self.layer1.weight,
                           self.layer2.weight,
                           self.layer3.weight,
                           self.layer4.weight
                          ]


    def forward(self, x):
        """
        Internal function representing the forward model.
        """
        x = self.layer0(x)
        x = self.activation(x)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.layer4(x)
        return x


    def fit(self, x_values, y_values, epochs=100, learning_rate=1e-5):
        """
        Function to train the weights of the multi layer perceptron.

        Parameters
        ----------
        x_values        : torch.tensor
                          Input values [mx1].
        y_values        : torch.tensor
                          Output values [nx1].
        epochs          : int
                          Number of epochs.
        learning_rate   : float
                          Learning rate of the optimizer.
        """
        t = tqdm(range(epochs), leave=False)
        self.optimizer = torch.optim.Adam(self.parameters, lr=learning_rate)
        for i in t:
            self.optimizer.zero_grad()
            estimate = self.forward(x_values)
            loss = self.loss_function(estimate, y_values)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            description = 'Loss:{:.4f}'.format(loss.item())
            t.set_description(description)
        print(description)
        return True


    def to(self, device):
        """
        Utilization function for setting the device.
        """
        self.device = device
        return self
