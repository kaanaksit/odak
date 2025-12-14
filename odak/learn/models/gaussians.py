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
    initial_centers    : torch.Tensor or None, optional
                         Initial centers of the Gaussians (shape: [N, 3]). If not provided,
                         random initialization is used where N is `number_of_elements`.
    initial_angles     : torch.Tensor or None, optional
                         Initial angles defining the orientation of each Gaussian. If not 
                         provided, random initialization is used.
    initial_scales     : torch.Tensor or None, optional
                         Initial scales controlling the spread (variance) of each Gaussian. 
                         If not provided, random initialization is used.
    initial_alphas     : torch.Tensor or None, optional
                         Initial alphas controlling the blending between Gaussians.
                         If not provided, random initialization is used.                         
    """    
    def __init__(
                 self,
                 number_of_elements = 10,
                 initial_centers = None,
                 initial_angles = None,
                 initial_scales = None,
                 initial_alphas = None,
                ):
        super(gaussian_3d_volume, self).__init__()
        self.number_of_elements = number_of_elements
        self.initialize_parameters(
                                   centers = initial_centers,
                                   angles = initial_angles,
                                   scales = initial_scales,
                                   alphas = initial_alphas,
                                  )
        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()


    def initialize_parameters(
                              self,
                              centers = None,
                              angles = None,
                              scales = None,
                              alphas = None,
                              device = torch.device('cpu'),
                             ):
        """
        Initialize model parameters using PyTorch tensors.

        Parameters
        ----------
        centers : torch.Tensor, optional
                  If None (default), initializes as a tensor of shape 
                  (number_of_elements, 3) with values sampled from standard normal distribution.
        angles  : torch.Tensor, optional
                  If None (default), initializes similarly to centers: shape (n,3).
        scales  : torch.Tensor, optional
                  If None (default), initializes as a tensor of shape 
                  (number_of_elements, 3) with values uniformly distributed between 0 and 1.
        alphas  : torch.Tensor, optional
                  If None (default), initializes as a tensor of shape 
                  (number_of_elements, 1) with values uniformly distributed between 0 and 1.        
        device  : torch.device
                  Device to be used to define the parameters.
                  Make sure to pass the device you use with this model for proper manual parameter initilization.
        """
        if isinstance(centers, type(None)):
            centers = torch.randn(self.number_of_elements, 3, device = device)
        if isinstance(angles, type(None)):
            angles = torch.randn(self.number_of_elements, 3, device = device)
        if isinstance(scales, type(None)):
            scales = torch.rand(self.number_of_elements, 3, device = device)
        if isinstance(alphas, type(None)):
            alphas = torch.rand(self.number_of_elements, 1, device = device)
        self.centers = torch.nn.Parameter(centers)
        self.angles = torch.nn.Parameter(angles)
        self.scales = torch.nn.Parameter(scales)
        self.alphas = torch.nn.Parameter(alphas)


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
                                            scales = self.scales,
                                            angles = self.angles * 180,
                                            opacity = self.alphas,
                                           )
        total_intensities = torch.mean(intensities, axis = -1)
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
                                 epoch_id = epoch_id,
                                 epoch_count = number_of_epochs,
                                 weights = loss_weights,
                                )
            loss.backward(retain_graph = True)
            optimizer.step()
            scheduler.step()
            description = 'gaussian_3d_volume model loss:{:.4f}'.format(loss.item())
            t_epoch.set_description(description)
            if epoch_id % save_at_every == save_at_every - 1:
                self.save_weights(weights_filename)
        logger.info(description)


    def evaluate(
                 self,
                 estimate,
                 ground_truth,
                 epoch_id = 0,
                 epoch_count = 1,
                 weights = {
                            'content'     : {
                                             'l2'         : 1e+0,
                                             'l1'         : 0e-0,
                                            },
                            'alpha'       : {
                                             'smaller'    : 0e-0,
                                             'larger'     : 0e-0,
                                             'threshold'  : [0., 1.]
                                            },                 
                            'scale'       : {
                                             'smaller'    : 0e-0,
                                             'larger'     : 0e-0,
                                             'threshold'  : [0., 1.],
                                            },
                            'alpha'       : 0e-0,
                            'angle'       : 0e-0,
                            'center'      : 0e-0, 
                            'utilization' : {
                                             'l2'         : 0e+0,
                                             'percentile' : 0
                                            }
                           },
                ):
        """
        Parameters
        ----------
        estimate     : torch.Tensor
                       Model's output estimate.
        ground_truth : torch.Tensor
                       Ground truth values.
        epoch_id     : int, optional
                       ID of the starting epoch. Default: 0.
        epoch_count  : int, optional
                       Total number of epochs for training. Default: 1.
        weights      : dict, optional
                       Dictionary containing weights for various loss components:
                       - content: {'l2': float, 'l1': float}
                       - scale: {'smaller': float, 'larger': float, 'threshold': List[float]}
                       - alpha: {'smaller': float, 'larger': float, 'threshold': List[float]}
                       - angle : float
                       - center: float
                       - utilization: {'l2': float, 'percentile': int}        
        """
        loss = 0.
        if weights['content']['l2'] != 0.:
            loss_l2_content = self.l2_loss(estimate, ground_truth)
            loss += weights['content']['l2'] * loss_l2_content
        if weights['content']['l1'] != 0.:
            loss_l1_content = self.l1_loss(estimate, ground_truth)
            loss += weights['content']['l1'] * loss_l1_content
        if weights['scale']['smaller'] != 0.:
            threshold = weights['scale']['threshold'][0]
            loss_scales_smaller = torch.sum(torch.abs(self.scales[self.scales < threshold]))
            loss += loss_scales_smaller * weights['scale']['smaller']
        if weights['scale']['larger'] != 0.:
            threshold = weights['scale']['threshold'][1]
            loss_scales_larger = torch.sum(self.scales[self.scales > threshold])
            loss += loss_scales_larger * weights['scale']['larger']
        if weights['alpha']['smaller'] != 0.:
            threshold = weights['alpha']['threshold'][0]
            loss_alphas_smaller = torch.sum(torch.abs(self.alphas[self.alphas < threshold]))
            loss += loss_alphas_smaller * weights['alpha']['smaller']
        if weights['alpha']['larger'] != 0.:
            threshold = weights['alpha']['threshold'][1]
            loss_alphas_larger = torch.sum(self.alphas[self.alphas > threshold])
            loss += loss_alphas_larger * weights['alpha']['larger']
        if weights['angle'] != 0.:
            loss_angle = torch.sum(self.angles[self.angles > 1.]) + \
                         torch.sum(torch.abs(self.angles[self.angles < -1.]))
            loss += weights['angle'] * loss_angle
        if weights['center'] != 0.:
            centers = torch.abs(self.centers)
            loss_center = torch.sum(centers[centers > 1.0])
            loss += weights['center'] * loss_center
        if weights['utilization']['l2'] !=0:
            n = self.alphas.numel()
            k = int(weights['utilization']['percentile'] / 100. * n)
            _, low_indices = torch.topk(torch.abs(self.alphas), k, dim = 0, largest = False)
            _, high_indices = torch.topk(torch.abs(self.alphas), k, dim = 0, largest = True)
            loss_utilization = torch.abs(torch.std(self.centers[low_indices, 0]) - torch.std(self.centers[high_indices, 0])) + \
                               torch.abs(torch.std(self.centers[low_indices, 1]) - torch.std(self.centers[high_indices, 1])) + \
                               torch.abs(torch.std(self.centers[low_indices, 2]) - torch.std(self.centers[high_indices, 2])) + \
                               torch.abs(torch.mean(self.centers[low_indices, 0]) - torch.mean(self.centers[high_indices, 0])) + \
                               torch.abs(torch.mean(self.centers[low_indices, 1]) - torch.mean(self.centers[high_indices, 1])) + \
                               torch.abs(torch.mean(self.centers[low_indices, 2]) - torch.mean(self.centers[high_indices, 2])) + \
                               torch.abs(torch.std(self.scales[low_indices, 0]) - torch.std(self.scales[high_indices, 0])) + \
                               torch.abs(torch.std(self.scales[low_indices, 1]) - torch.std(self.scales[high_indices, 1])) + \
                               torch.abs(torch.std(self.scales[low_indices, 2]) - torch.std(self.scales[high_indices, 2])) + \
                               torch.abs(torch.mean(self.scales[low_indices, 0]) - torch.mean(self.scales[high_indices, 0])) + \
                               torch.abs(torch.mean(self.scales[low_indices, 1]) - torch.mean(self.scales[high_indices, 1])) + \
                               torch.abs(torch.mean(self.scales[low_indices, 2]) - torch.mean(self.scales[high_indices, 2])) + \
                               torch.abs(torch.mean(self.alphas[low_indices]) - torch.mean(self.alphas[high_indices])) + \
                               torch.abs(torch.std(self.alphas[low_indices]) - torch.std(self.alphas[high_indices]))
            loss_distribution = torch.std(self.centers[:, 0]) + \
                                torch.std(self.centers[:, 1]) + \
                                torch.std(self.centers[:, 2]) + \
                                torch.std(self.scales[:, 0]) +\
                                torch.std(self.scales[:, 1]) +\
                                torch.std(self.scales[:, 2]) +\
                                torch.std(self.alphas)
            decay = 1. - ((epoch_count - epoch_id) / epoch_count)
            loss += decay * weights['utilization']['l2'] * \
                    (loss_distribution + loss_utilization)
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
            weights_filename = expanduser(weights_filename)
            if os.path.isfile(weights_filename):
                self.load_state_dict(
                                     torch.load(
                                                weights_filename,
                                                weights_only = True,
                                                map_location = device
                                               )
                                   )
                self.eval()
                logger.info('gaussian_3d_volume model weights loaded: {}'.format(weights_filename))
