from ..tools import evaluate_3d_gaussians, expanduser
from ...log import logger
import torch
import os
from tqdm import tqdm


class gaussian_3d_volume(torch.nn.Module):
    """
    Initialize the 3D Gaussian volume model. This model is useful for learning voxelized 3D volumes.

    Parameters
    ----------
    number_of_elements : int
                         Number of Gaussian elements in the volume (default: 10).
    """    
    def __init__(
                 self,
                 number_of_elements = 10,
                ):
        super(gaussian_3d_volume, self).__init__()
        self.centers = torch.nn.Parameter(
                                          torch.rand(number_of_elements, 3)
                                         )
        self.angles = torch.nn.Parameter(
                                         torch.randn(number_of_elements, 3)
                                        )
        self.sigmas = torch.nn.Parameter(
                                         torch.rand(number_of_elements, 3)
                                        )
        self.alphas = torch.nn.Parameter(
                                         torch.rand(number_of_elements, 1)
                                        )
        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()


    def forward(self, points, test = False):
        """
        Forward pass: evaluate the 3D Gaussian volume at given points.

        Parameters
        ----------
        points            : torch.Tensor,  shape (N, 3)
                            Input points at which to evaluate the Gaussian volume, where each row is a 3D point.
        test              : bool, optional
                            If True, disables gradient computation (default: False).

        Returns
        -------
        total_intensities : torch.Tensor
                            Total intensities at the input points, weighted by alphas.
        """        
        if test:
            torch.no_grad()
        intensities = evaluate_3d_gaussians(
                                            points = points,
                                            centers = self.centers,
                                            sigmas = self.sigmas,
                                            angles = self.angles * 180,
                                            opacity = self.alphas,
                                           )
        total_intensities = torch.sum(intensities, axis = -1)
        return total_intensities


    def optimize(
                 self,
                 points,
                 ground_truth,
                 loss_weights,
                 learning_rate = 1e-2,
                 number_of_epochs = 10,
                 scheduler_power = 1,
                 save_at_every = 1,
                 weights_filename = None,
                ):
        """
        Optimize model parameters using AdamW and a polynomial learning rate scheduler.

        Parameters
        ----------
        points           : torch.Tensor
                           Input data points for the model.
        ground_truth     : torch.Tensor
                           Ground truth values corresponding to the input points.
        loss_weights     : dict
                           Dictionary of weights for each loss component.
        learning_rate    : float, optional
                           Learning rate for the optimizer. Default is 1e-2.
        number_of_epochs : int, optional
                           Number of training epochs. Default is 10.
        scheduler_power  : float, optional
                           Power parameter for the polynomial learning rate scheduler. Default is 1.
        save_at_every    : int
                           Save model weights every `save_at_every` epochs. Default is 1.
        weights_filename : str, optional
                           Filename for saving model weights. If None, weights are not saved.

        Notes
        -----
        - Uses AdamW optimizer and PolynomialLR scheduler.
        - Logs loss at each epoch and saves weights periodically.
        """                
        optimizer = torch.optim.AdamW(self.parameters(), lr = learning_rate)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
                                                          optimizer,
                                                          total_iters = number_of_epochs,
                                                          power = scheduler_power,
                                                          last_epoch = -1
                                                         )
        t_epoch = tqdm(range(number_of_epochs), leave = False, dynamic_ncols = True)
        for epoch_id in t_epoch:
            optimizer.zero_grad()
            estimates = self.forward(points)
            loss = self.evaluate(
                                 estimates,
                                 ground_truth,
                                 weights = loss_weights,
                                )
            loss.backward(retain_graph = True)
            optimizer.step()
            scheduler.step()
            description = 'Loss:{:.4f}'.format(loss.item())
            t_epoch.set_description(description)
            if epoch_id % save_at_every == save_at_every - 1:
                self.save_weights(weights_filename)
        logger.info(description)


    def evaluate(
                 self,
                 estimate,
                 ground_truth,
                 weights = {
                            'content' : {
                                         'l2'  : 1e+0,
                                         'l1'  : 0e-0,
                                        },
                            'sigma'   : 0e-0,
                            'alpha'   : 0e-0,
                            'angle'   : 0e-0,
                            'center'  : 0e-0, 
                           },
                ):
        """
        Evaluate the model's loss using weighted combinations of L1, L2, and regularization terms.

        Parameters
        ----------
        estimate     : torch.Tensor
                       Model's output estimate.
        ground_truth : torch.Tensor
                       Ground truth values.
        weights      : dict, optional
                       Dictionary of weights for each loss component.

        Returns
        -------
        loss         : torch.Tensor
                       Computed loss value.

        Notes
        -----
        - Loss is a weighted sum of L1, L2, and regularization terms for sigma, alpha, and angle.
        - Only non-zero weights are used in the loss calculation.
        """
        loss = 0.
        if weights['content']['l2'] != 0.:
            loss_l2_content = self.l2_loss(estimate, ground_truth)
            loss += weights['content']['l2'] * loss_l2_content
        if weights['content']['l1'] != 0.:
            loss_l1_content = self.l1_loss(estimate, ground_truth)
            loss += weights['content']['l1'] * loss_l1_content
        if weights['sigma'] != 0.:
            loss_sigmas = torch.sum(torch.abs(self.sigmas)[self.sigmas < 1e-2])
            loss += weights['sigma'] * loss_sigmas
        if weights['alpha'] != 0.:
            loss_alphas = torch.sum(self.alphas[self.alphas > 1.]) + \
                          torch.sum(torch.abs(self.alphas[self.alphas < 1e-2]))
            loss += weights['alpha'] * loss_alphas
        if weights['angle'] != 0.:
            loss_angle = torch.sum(self.angles[self.angles > 1.]) + \
                         torch.sum(torch.abs(self.angles[self.angles < -1.]))
            loss += weights['angle'] * loss_angle
        if weights['center'] != 0.:
            loss_center = torch.sum(
                                    torch.abs(
                                              self.centers[torch.abs(self.centers) > 1.0]
                                             )
                                   )
            loss += weights['center'] * loss_center
        return loss


    def save_weights(self, weights_filename):
        """
        Save the model weights to a specified file.


        Parameters
        ----------
        weights_filename : str
                           Path or filename where the weights will be saved. The path can include 
                           relative paths and tilde notation (~), which will be expanded by `expanduser`.


        Example:
        --------
        # Save model weights to current directory with filename 'model_weights.pth'
        save_weights('model_weights.pth')

        # Save model weights to home directory using ~ notation
        save_weights('~/.weights.pth')
        """        
        weights_filename = expanduser(weights_filename)
        torch.save(self.state_dict(), weights_filename)
        logger.info('gaussian_3d_volume model weights saved: {}'.format(weights_filename))


    def load_weights(self, weights_filename = None, device = torch.device('cpu')):
        """
        Load model weights from a file.

        Parameters
        ----------
        weights_filename : str
                           Path to the weights file. If None, no weights are loaded.
        device           : torch.device, optional
                           Device to load the weights onto (default: 'cpu').

        Notes
        -----
        - If `weights_filename` is a valid file, the model state is updated and set to eval mode.
        - The file path is expanded (e.g., '~' is resolved).
        - A log message is emitted upon successful loading.
        """
        if not isinstance(weights_filename, type(None)):
            if os.path.isfile(weights_filename):
                weights_filename = expanduser(weights_filename)
                self.load_state_dict(
                                     torch.load(
                                                weights_filename,
                                                weights_only = True,
                                                map_location = device
                                               )
                                   )
                self.eval()
                logger.info('gaussian_3d_volume model weights loaded: {}'.format(weights_filename))
