import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class multi_layer_perceptron(nn.Module):
    """
    A flexible class to correlate one dimensional input data to an one dimensional output data.
    """
    def __init__(self, n_input=1, n_hidden=64, n_output=1, n_layers=4):
        """
        Parameters
        ----------
        n_input         : int
                          Input size [a].
        n_hidden        : int
                          Hidden layer size [b].
        n_output        : int
                          Output size [c].
        n_layers        : int
                          Number of cascaded linear layers.
        """
        super(multi_layer_perceptron, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(torch.nn.Linear(n_input, n_hidden))
        for i in range(n_layers):
            new_layer = nn.Sequential(
                                      torch.nn.Linear(n_hidden, n_hidden),
                                      torch.nn.ReLU(inplace=True)
                                     )
            self.layers.append(new_layer)
        self.layer_final = torch.nn.Linear(n_hidden, n_output)


    def forward(self, x):
        """
        Internal function representing the forward model.
        """
        for layer in self.layers:
            x = layer(x)
        x = self.layer_final(x)
        return x

    def estimate(self, x):
        """
        Internal function representing the forward model w/o grad.
        """
        return self.forward(x).detach()


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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_function = torch.nn.MSELoss()
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


    def save_weights(self, filename='./weights.pt'):
        """
        Function to save the current weights of the multi layer perceptron to a file.

        Parameters
        ----------
        filename        : str
                          Filename.
        """
        torch.save(self.state_dict(), filename)


    def load_weights(self, filename='./weights.pt'):
        """
        Function to load weights for this multi layer perceptron from a file.

        Parameters
        ----------
        filename        : str
                          Filename.
        """
        self.load_state_dict(torch.load(filename))

