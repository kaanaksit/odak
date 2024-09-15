import torch
import os
from tqdm import tqdm
from ..models import unet
from .util import generate_complex_field, wavenumber


class holobeam_multiholo(torch.nn.Module):
    """
    The learned holography model used in the paper, Akşit, Kaan, and Yuta Itoh. "HoloBeam: Paper-Thin Near-Eye Displays." In 2023 IEEE Conference Virtual Reality and 3D User Interfaces (VR), pp. 581-591. IEEE, 2023.


    Parameters
    ----------
    n_input           : int
                        Number of channels in the input.
    n_hidden          : int
                        Number of channels in the hidden layers.
    n_output          : int
                        Number of channels in the output layer.
    device            : torch.device
                        Default device is CPU.
    reduction         : str
                        Reduction used for torch.nn.MSELoss and torch.nn.L1Loss. The default is 'sum'.
    """
    def __init__(
                 self,
                 n_input = 1,
                 n_hidden = 16,
                 n_output = 2,
                 device = torch.device('cpu'),
                 reduction = 'sum'
                ):
        super(holobeam_multiholo, self).__init__()
        torch.random.seed()
        self.device = device
        self.reduction = reduction
        self.l2 = torch.nn.MSELoss(reduction = self.reduction)
        self.l1 = torch.nn.L1Loss(reduction = self.reduction)
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.network = unet(
                            dimensions = self.n_hidden,
                            input_channels = self.n_input,
                            output_channels = self.n_output
                           ).to(self.device)


    def forward(self, x, test = False):
        """
        Internal function representing the forward model.
        """
        if test:
            torch.no_grad()
        y = self.network.forward(x) 
        phase_low = y[:, 0].unsqueeze(1)
        phase_high = y[:, 1].unsqueeze(1)
        phase_only = torch.zeros_like(phase_low)
        phase_only[:, :, 0::2, 0::2] = phase_low[:, :,  0::2, 0::2]
        phase_only[:, :, 1::2, 1::2] = phase_low[:, :, 1::2, 1::2]
        phase_only[:, :, 0::2, 1::2] = phase_high[:, :, 0::2, 1::2]
        phase_only[:, :, 1::2, 0::2] = phase_high[:, :, 1::2, 0::2]
        return phase_only


    def evaluate(self, input_data, ground_truth, weights = [1., 0.1]):
        """
        Internal function for evaluating.
        """
        loss = weights[0] * self.l2(input_data, ground_truth) + weights[1] * self.l1(input_data, ground_truth)
        return loss


    def fit(self, dataloader, number_of_epochs = 100, learning_rate = 1e-5, directory = './output', save_at_every = 100):
        """
        Function to train the weights of the multi layer perceptron.

        Parameters
        ----------
        dataloader       : torch.utils.data.DataLoader
                           Data loader.
        number_of_epochs : int
                           Number of epochs.
        learning_rate    : float
                           Learning rate of the optimizer.
        directory        : str
                           Output directory.
        save_at_every    : int
                           Save the model at every given epoch count.
        """
        t_epoch = tqdm(range(number_of_epochs), leave=False, dynamic_ncols = True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for i in t_epoch:
            epoch_loss = 0.
            t_data = tqdm(dataloader, leave=False, dynamic_ncols = True)
            for j, data in enumerate(t_data):
                self.optimizer.zero_grad()
                images, holograms = data
                estimates = self.forward(images)
                loss = self.evaluate(estimates, holograms)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                description = 'Loss:{:.4f}'.format(loss.item())
                t_data.set_description(description)
                epoch_loss += float(loss.item()) / dataloader.__len__()
            description = 'Epoch Loss:{:.4f}'.format(epoch_loss)
            t_epoch.set_description(description)
            if i % save_at_every == 0:
                self.save_weights(filename='{}/weights_{:04d}.pt'.format(directory, i))
        self.save_weights(filename='{}/weights.pt'.format(directory))
        print(description)

    
    def save_weights(self, filename = './weights.pt'):
        """
        Function to save the current weights of the multi layer perceptron to a file.
        Parameters
        ----------
        filename        : str
                          Filename.
        """
        torch.save(self.network.state_dict(), os.path.expanduser(filename))


    def load_weights(self, filename = './weights.pt'):
        """
        Function to load weights for this multi layer perceptron from a file.
        Parameters
        ----------
        filename        : str
                          Filename.
        """
        self.network.load_state_dict(torch.load(os.path.expanduser(filename)))
        self.network.eval()
