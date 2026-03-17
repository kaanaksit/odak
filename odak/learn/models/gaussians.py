from ..tools import evaluate_3d_gaussians
from ...tools.file import validate_path
from ...log import logger
import torch
import os
from pathlib import Path
from tqdm import tqdm


class gaussian_2d(torch.nn.Module):
    """
    2D Gaussian model for learning image representations using 2D Gaussian primitives.

    This model represents an image as a weighted sum of 2D Gaussians, each defined by:
    - widths (std_x, std_y): Standard deviations along x and y axes
    - offsets (offset_x, offset_y): Center positions in normalized coordinates
    - rotations: Rotation angles for each Gaussian
    - alphas: Opacity/weight coefficients

    Parameters
    ----------
    number_of_elements : int, optional
                        Number of 2D Gaussian elements to use. Default is 10.

    Attributes
    ----------
    widths      : torch.nn.Parameter, shape (2, 1, N)
                  Standard deviations for x and y dimensions.
    offsets     : torch.nn.Parameter, shape (2, 1, N)
                  Center offsets in x and y directions.
    rotations   : torch.nn.Parameter, shape (1, N)
                  Rotation angles in radians for each Gaussian.
    alphas      : torch.nn.Parameter, shape (1, N)
                  Opacity/weight coefficients blended with tanh activation.

    Examples
    --------
    >>> model = gaussian_2d(number_of_elements=50)
    >>> x = torch.linspace(-1, 1, 256)
    >>> y = torch.linspace(-1, 1, 256)
    >>> X, Y = torch.meshgrid(x, y, indexing='ij')
    >>> output = model(X, Y)

    Notes
    -----
    - All parameters are initialized on CPU by default. For GPU acceleration,
      call .to(device) after initializing this model.
    - Input coordinates x and y should typically be normalized to [-1, 1].
    - Output is the sum of weighted Gaussians passed through tanh().
    """

    def __init__(self, number_of_elements=10):
        """
        Initialize the 2D Gaussian model.

        Parameters
        ----------
        number_of_elements : int
                            Number of Gaussian elements (default: 10).
        """
        super(gaussian_2d, self).__init__()

        if not isinstance(number_of_elements, int) or number_of_elements <= 0:
            raise ValueError(
                "number_of_elements must be a positive integer, got {}".format(
                    type(number_of_elements).__name__
                )
            )

        self.number_of_elements = number_of_elements

        # Initialize parameters as learnable tensors
        self.widths = torch.nn.Parameter(torch.rand(2, 1, self.number_of_elements))
        self.offsets = torch.nn.Parameter(
            torch.randn(2, 1, self.number_of_elements)
        )
        self.rotations = torch.nn.Parameter(torch.randn(1, self.number_of_elements))
        self.alphas = torch.nn.Parameter(torch.randn(1, self.number_of_elements))

        # Apply uniform initialization
        self.initialize_parameters_uniformly()

    def initialize_parameters_uniformly(self, ranges=None):
        """
        Initialize parameters using uniform-like distributions within specified ranges.

        This method re-samples the model parameters from normal distributions
        whose mean and standard deviation are derived from the provided ranges.
        For a range [a, b], it uses:
            mean = (a + b) / 2
            std  = (b - a) / 4

        Parameters
        ----------
        ranges : dict or None, optional
                Dictionary specifying custom initialization ranges. Keys can include:
                - 'widths': tuple of (min, max) for Gaussian widths
                - 'offsets': tuple of (min, max) for center offsets
                - 'rotations': tuple of (min, max) for rotation angles in radians
                - 'alphas': tuple of (min, max) for opacity values

                If None, default ranges are used:
                {
                    "widths": (0.1, 0.5),
                    "offsets": (-1.0, 1.0),
                    "rotations": (0.0, 2*pi),
                    "alphas": (0.1, 0.2)
                }

        Notes
        -----
        - Uses torch.no_grad() to avoid tracking gradients during initialization.
        - Parameters are initialized in-place using normal_() method.
        """
        with torch.no_grad():
            default_ranges = {
                "widths": (0.1, 0.5),
                "offsets": (-1.0, 1.0),
                "rotations": (0.0, 2 * torch.pi),
                "alphas": (0.1, 0.2),
            }

            if ranges is None:
                ranges = default_ranges

            # Initialize widths (std_x and std_y)
            if "widths" in ranges:
                self.widths.normal_(
                    mean=(ranges["widths"][0] + ranges["widths"][1]) / 2,
                    std=(ranges["widths"][1] - ranges["widths"][0]) / 4,
                )
            else:
                self.widths.normal_(mean=0.3, std=0.1)

            # Initialize offsets (offset_x and offset_y)
            if "offsets" in ranges:
                self.offsets.normal_(
                    mean=(ranges["offsets"][0] + ranges["offsets"][1]) / 2,
                    std=(ranges["offsets"][1] - ranges["offsets"][0]) / 4,
                )
            else:
                self.offsets.normal_(mean=0.0, std=0.5)

            # Initialize rotations
            if "rotations" in ranges:
                self.rotations.normal_(
                    mean=(ranges["rotations"][0] + ranges["rotations"][1]) / 2,
                    std=(ranges["rotations"][1] - ranges["rotations"][0]) / 4,
                )
            else:
                self.rotations.normal_(mean=torch.pi, std=torch.pi / 2)

            # Initialize alphas (opacity coefficients)
            if "alphas" in ranges:
                self.alphas.normal_(
                    mean=(ranges["alphas"][0] + ranges["alphas"][1]) / 2,
                    std=(ranges["alphas"][1] - ranges["alphas"][0]) / 4,
                )
            else:
                self.alphas.normal_(mean=0.15, std=0.05)

    def forward(self, x, y, residual=1e-6):
        """
        Forward pass: evaluate the 2D Gaussian model at given coordinates.

        Computes a weighted sum of 2D Gaussians evaluated at the input grid
        coordinates (x, y). Each Gaussian is rotated and translated according
        to its learned parameters.

        Parameters
        ----------
        x : torch.Tensor
            X-coordinates of the evaluation grid. Shape should broadcast with y.
        y : torch.Tensor
            Y-coordinates of the evaluation grid. Shape should broadcast with x.
        residual : float, optional
                   Small constant to avoid numerical issues (default: 1e-6).

        Returns
        -------
        results : torch.Tensor
                  The evaluated Gaussian field at input coordinates. The output
                  shape is determined by broadcasting x, y with the parameter shapes.
                  Values are passed through tanh() activation and multiplied by alphas.

        Notes
        -----
        - Coordinates are first rotated using learned rotation angles.
        - Then translated by learned offsets for each Gaussian.
        - The 2D Gaussian function is evaluated as exp(-(x^2 + y^2)) scaled by widths.
        - Final output: tanh(alphas * gaussians) summed over all elements.

    Notes
    -----
    - Supports multiple input shapes via PyTorch broadcasting
    - For grid inputs (H, W): automatically broadcasts to (H, W, N_elements)
    - For flattened inputs (N, 1): broadcasts directly with parameters
    """
        # PyTorch broadcasting handles shape alignment automatically
        # Input shapes: x, y can be (H, W), (H*W,), or (-1, 1)
        # Parameters are stored as (2, 1, N) for offsets/widths and (1, N) for rotations/alphas
        
        # Rotate coordinates according to each Gaussian's rotation angle
        cos_rot = torch.cos(self.rotations)  # Shape: (1, N)
        sin_rot = torch.sin(self.rotations)  # Shape: (1, N)
        
        # Broadcasting: x (*), y (*) automatically expand with cos_rot/sin_rot
        x_r = x * cos_rot - y * sin_rot
        y_r = x * sin_rot + y * cos_rot

        # Translate by learned offsets (broadcasts from (2, 1, N) to input shape × (N,))
        x_n = x_r + self.offsets[0]  # Shape: (..., N)
        y_n = y_r + self.offsets[1]

        # Evaluate 2D Gaussian function with learned widths (standard deviations)
        r = (x_n / self.widths[0]) ** 2 + (y_n / self.widths[1]) ** 2
        gaussians = torch.exp(-r)

        # Apply alpha weights and tanh activation
        results = self.alphas * gaussians
        results = torch.tanh(results)

        return results


class gaussians_2d(torch.nn.Module):
    """
    Wrapper class for the 2D Gaussian model with loss computation and evaluation utilities.

    This class wraps `gaussian_2d` and provides additional functionality:
    - Loss functions (L1, L2) pre-initialized
    - Weight saving/loading methods
    - Model parameter counting

    Parameters
    ----------
    number_of_elements : int, optional
                        Number of 2D Gaussian primitives in the model (default: 10).
    logger             : logging.Logger or None, optional
                         Logger instance for tracking progress. If None, creates a new one.

    Attributes
    ----------
    model       : gaussian_2d
                  The underlying primitive Gaussian model.
    l2_loss     : torch.nn.MSELoss
                  Mean squared error loss function.
    l1_loss     : torch.nn.L1Loss
                  L1 absolute loss function.
    logger      : logging.Logger
                  Logger instance for info/debug messages.

    Examples
    --------
    >>> model = gaussians_2d(number_of_elements=50)
    >>> x = torch.linspace(-1, 1, 256)
    >>> y = torch.linspace(-1, 1, 256)
    >>> X, Y = torch.meshgrid(x, y, indexing='ij')
    >>> output = model(X, Y, test=False)

    Notes
    -----
    - The `test` flag in forward() controls gradient computation (not recommended use).
    - Use standard training loop with optimizer.zero_grad(), loss.backward(), optimizer.step().
    """

    def __init__(
        self,
        number_of_elements=10,
        logger=None,
    ):
        """
        Initialize the gaussians_2d wrapper model.

        Parameters
        ----------
        number_of_elements : int
                            Number of 2D Gaussian elements (default: 10).
        logger             : logging.Logger or None
                             Logger instance (default: creates new logger).
        """
        super(gaussians_2d, self).__init__()

        if not isinstance(number_of_elements, int) or number_of_elements <= 0:
            raise ValueError(
                "number_of_elements must be a positive integer, got {}".format(
                    type(number_of_elements).__name__ if not isinstance(number_of_elements, int) else str(number_of_elements)
                )
            )

        self.number_of_elements = number_of_elements
        self.model = gaussian_2d(
            number_of_elements=self.number_of_elements
        )

        # Count total trainable parameters
        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Setup logger
        from ...log import logger as default_logger

        self.logger = logger if logger is not None else default_logger

    def forward(
        self,
        x,
        y,
        test=False,
    ):
        """
        Forward pass through the Gaussian model.

        Parameters
        ----------
        x      : torch.Tensor
                 X-coordinates of evaluation grid.
        y      : torch.Tensor
                 Y-coordinates of evaluation grid.
        test   : bool, optional
                 If True, runs in no_grad mode (default: False).

        Returns
        -------
        result : torch.Tensor
                 The summed Gaussian field with shape matching x/y grids plus one dimension.

        Notes
        -----
        The `test` flag is deprecated. Use standard training pattern:
        ```python
        model.train()  # Enable gradients
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        ```
        """
        if test:
            with torch.no_grad():
                result = self.model(x=x, y=y)
        else:
            result = self.model(x=x, y=y)

        # Sum over Gaussian elements and add batch dimension
        result = torch.sum(result, dim=-1).unsqueeze(-1)

        return result

    def save_weights(self, weights_filename):
        """
        Save model weights to a file.

        Parameters
        ----------
        weights_filename : str
                          Path to save weights (must end with .pt, .pth, or similar).
        """
        from ...tools.file import validate_path

        safe_path = validate_path(
            os.path.expanduser(weights_filename), 
            allowed_extensions=[".pt", ".pth"]
        )
        torch.save(self.state_dict(), safe_path)
        self.logger.info("Model weights saved to: {}".format(safe_path))

    def load_weights(
        self, 
        weights_filename=None,
        device=torch.device("cpu")
    ):
        """
        Load model weights from a file.

        Parameters
        ----------
        weights_filename : str or None
                          Path to weights file. If None, skips loading.
        device           : torch.device, optional
                          Device to load weights onto (default: CPU).
        """
        if weights_filename is not None:
            from ...tools.file import validate_path

            safe_path = validate_path(
                os.path.expanduser(weights_filename),
                allowed_extensions=[".pt", ".pth"]
            )

            if not os.path.isfile(safe_path):
                raise FileNotFoundError("Weights file not found: {}".format(safe_path))

            self.load_state_dict(
                torch.load(safe_path, weights_only=True, map_location=device)
            )
            self.eval()  # Set to evaluation mode
            self.logger.info("Model weights loaded from: {}".format(safe_path))


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
        number_of_elements=10,
        initial_centers=None,
        initial_angles=None,
        initial_scales=None,
        initial_alphas=None,
    ):
        """
        Initialize the 3D Gaussian volume model.

        Parameters
        ----------
        number_of_elements : int
                            Number of Gaussian elements in the volume (default: 10).
        initial_centers    : torch.Tensor or None
                            Initial centers of the Gaussians (shape: [N, 3]).
        initial_angles     : torch.Tensor or None
                            Initial angles for orientation.
        initial_scales     : torch.Tensor or None
                            Initial scales for variance.
        initial_alphas     : torch.Tensor or None
                            Initial alphas for blending.

        Device Placement
        ----------- --
        All parameters are initialized on CPU by default. For GPU acceleration,
        call .to(device) after initializing this model.
        Example:
            model = gaussian_3d_volume().cuda()  # or .to('cuda')
        """
        super(gaussian_3d_volume, self).__init__()
        self.number_of_elements = number_of_elements
        self.initialize_parameters(
            centers=initial_centers,
            angles=initial_angles,
            scales=initial_scales,
            alphas=initial_alphas,
        )
        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()

    def initialize_parameters(
        self,
        centers=None,
        angles=None,
        scales=None,
        alphas=None,
        device=torch.device("cpu"),
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
            centers = torch.randn(self.number_of_elements, 3, device=device)
        if isinstance(angles, type(None)):
            angles = torch.randn(self.number_of_elements, 3, device=device)
        if isinstance(scales, type(None)):
            scales = torch.rand(self.number_of_elements, 3, device=device)
        if isinstance(alphas, type(None)):
            alphas = torch.rand(self.number_of_elements, 1, device=device)
        self.centers = torch.nn.Parameter(centers)
        self.angles = torch.nn.Parameter(angles)
        self.scales = torch.nn.Parameter(scales)
        self.alphas = torch.nn.Parameter(alphas)

    def forward(self, points, test=False):
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
            points=points,
            centers=self.centers,
            scales=self.scales,
            angles=self.angles * 180,
            opacity=self.alphas,
        )
        total_intensities = torch.mean(intensities, axis=-1)
        return total_intensities

    def optimize(
        self,
        points,
        ground_truth,
        loss_weights,
        learning_rate=1e-2,
        number_of_epochs=10,
        scheduler_power=1,
        save_at_every=1,
        max_norm=None,
        weights_filename=None,
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
        max_norm         : float, optional
                           By default it is None, when set clips the gradient with the given threshold.
        weights_filename : str, optional
                           Filename for saving model weights. If None, weights are not saved.

        Notes
        -----
        - Uses AdamW optimizer and PolynomialLR scheduler.
        - Logs loss at each epoch and saves weights periodically.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=number_of_epochs,
            power=scheduler_power,
            last_epoch=-1,
        )
        t_epoch = tqdm(range(number_of_epochs), leave=False, dynamic_ncols=True)
        for epoch_id in t_epoch:
            optimizer.zero_grad()
            estimates = self.forward(points)
            loss = self.evaluate(
                estimates,
                ground_truth,
                epoch_id=epoch_id,
                epoch_count=number_of_epochs,
                weights=loss_weights,
            )
            loss.backward(retain_graph=True)
            if not isinstance(max_norm, type(None)):
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
            optimizer.step()
            scheduler.step()
            description = "gaussian_3d_volume model loss:{:.4f}".format(loss.item())
            t_epoch.set_description(description)
            if epoch_id % save_at_every == save_at_every - 1:
                self.save_weights(weights_filename)
        logger.info(description)

    def evaluate(
        self,
        estimate,
        ground_truth,
        epoch_id=0,
        epoch_count=1,
        weights={
            "content": {
                "l2": 1e0,
                "l1": 0e-0,
            },
            "alpha": {"smaller": 0e-0, "larger": 0e-0, "threshold": [0.0, 1.0]},
            "scale": {
                "smaller": 0e-0,
                "larger": 0e-0,
                "threshold": [0.0, 1.0],
            },
            "alpha": 0e-0,
            "angle": 0e-0,
            "center": 0e-0,
            "utilization": {"l2": 0e0, "percentile": 0},
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
        loss = 0.0
        if weights["content"]["l2"] != 0.0:
            loss_l2_content = self.l2_loss(estimate, ground_truth)
            loss += weights["content"]["l2"] * loss_l2_content
        if weights["content"]["l1"] != 0.0:
            loss_l1_content = self.l1_loss(estimate, ground_truth)
            loss += weights["content"]["l1"] * loss_l1_content
        if weights["scale"]["smaller"] != 0.0:
            threshold = weights["scale"]["threshold"][0]
            loss_scales_smaller = torch.sum(
                torch.abs(self.scales[self.scales < threshold])
            )
            loss += loss_scales_smaller * weights["scale"]["smaller"]
        if weights["scale"]["larger"] != 0.0:
            threshold = weights["scale"]["threshold"][1]
            loss_scales_larger = torch.sum(self.scales[self.scales > threshold])
            loss += loss_scales_larger * weights["scale"]["larger"]
        if weights["alpha"]["smaller"] != 0.0:
            threshold = weights["alpha"]["threshold"][0]
            loss_alphas_smaller = torch.sum(
                torch.abs(self.alphas[self.alphas < threshold])
            )
            loss += loss_alphas_smaller * weights["alpha"]["smaller"]
        if weights["alpha"]["larger"] != 0.0:
            threshold = weights["alpha"]["threshold"][1]
            loss_alphas_larger = torch.sum(self.alphas[self.alphas > threshold])
            loss += loss_alphas_larger * weights["alpha"]["larger"]
        if weights["angle"] != 0.0:
            loss_angle = torch.sum(self.angles[self.angles > 1.0]) + torch.sum(
                torch.abs(self.angles[self.angles < -1.0])
            )
            loss += weights["angle"] * loss_angle
        if weights["center"] != 0.0:
            centers = torch.abs(self.centers)
            loss_center = torch.sum(centers[centers > 1.0])
            loss += weights["center"] * loss_center
        if weights["utilization"]["l2"] != 0:
            n = self.alphas.numel()
            k = int(weights["utilization"]["percentile"] / 100.0 * n)
            _, low_indices = torch.topk(torch.abs(self.alphas), k, dim=0, largest=False)
            _, high_indices = torch.topk(torch.abs(self.alphas), k, dim=0, largest=True)
            loss_utilization = (
                torch.abs(
                    torch.std(self.centers[low_indices, 0])
                    - torch.std(self.centers[high_indices, 0])
                )
                + torch.abs(
                    torch.std(self.centers[low_indices, 1])
                    - torch.std(self.centers[high_indices, 1])
                )
                + torch.abs(
                    torch.std(self.centers[low_indices, 2])
                    - torch.std(self.centers[high_indices, 2])
                )
                + torch.abs(
                    torch.mean(self.centers[low_indices, 0])
                    - torch.mean(self.centers[high_indices, 0])
                )
                + torch.abs(
                    torch.mean(self.centers[low_indices, 1])
                    - torch.mean(self.centers[high_indices, 1])
                )
                + torch.abs(
                    torch.mean(self.centers[low_indices, 2])
                    - torch.mean(self.centers[high_indices, 2])
                )
                + torch.abs(
                    torch.std(self.scales[low_indices, 0])
                    - torch.std(self.scales[high_indices, 0])
                )
                + torch.abs(
                    torch.std(self.scales[low_indices, 1])
                    - torch.std(self.scales[high_indices, 1])
                )
                + torch.abs(
                    torch.std(self.scales[low_indices, 2])
                    - torch.std(self.scales[high_indices, 2])
                )
                + torch.abs(
                    torch.mean(self.scales[low_indices, 0])
                    - torch.mean(self.scales[high_indices, 0])
                )
                + torch.abs(
                    torch.mean(self.scales[low_indices, 1])
                    - torch.mean(self.scales[high_indices, 1])
                )
                + torch.abs(
                    torch.mean(self.scales[low_indices, 2])
                    - torch.mean(self.scales[high_indices, 2])
                )
                + torch.abs(
                    torch.mean(self.alphas[low_indices])
                    - torch.mean(self.alphas[high_indices])
                )
                + torch.abs(
                    torch.std(self.alphas[low_indices])
                    - torch.std(self.alphas[high_indices])
                )
            )
            loss_distribution = (
                torch.std(self.centers[:, 0])
                + torch.std(self.centers[:, 1])
                + torch.std(self.centers[:, 2])
                + torch.std(self.scales[:, 0])
                + torch.std(self.scales[:, 1])
                + torch.std(self.scales[:, 2])
                + torch.std(self.alphas)
            )
            decay = 1.0 - ((epoch_count - epoch_id) / epoch_count)
            loss += (
                decay
                * weights["utilization"]["l2"]
                * (loss_distribution + loss_utilization)
            )
        return loss

    def save_weights(self, weights_filename):
        """
        Save the model weights to a specified file.


        Parameters
        ----------
        weights_filename : str
                            Path or filename where the weights will be saved. The path can include
                            relative paths and tilde notation (~), which will be expanded by `validate_path`.


        Example:
        --------
        # Save model weights to current directory with filename 'model_weights.pth'
        save_weights('model_weights.pth')

        # Save model weights to home directory using ~ notation
        save_weights('~/.weights.pth')

        Raises
        ------
        ValueError : If path validation fails or extension is not allowed.
        """
        safe_path = validate_path(
            weights_filename, allowed_extensions=[".pth", ".pt", ".bin"]
        )
        torch.save(self.state_dict(), safe_path)
        logger.info("gaussian_3d_volume model weights saved: {}".format(safe_path))

    def load_weights(self, weights_filename=None, device=torch.device("cpu")):
        """
        Load model weights from a file.

        Parameters
        ----------
        weights_filename : str
                            Path to the weights file. If None, no weights are loaded.
        device           : torch.device, optional
                            Device to load the weights onto (default: 'cpu').

        Raises
        ------
        ValueError       : If path validation fails or extension is not allowed.
        FileNotFoundError: If file does not exist after validation.

        Notes
        -----
        - If `weights_filename` is a valid file, the model state is updated and set to eval mode.
        - The file path is validated for security (tilde expanded, path traversal blocked).
        - A log message is emitted upon successful loading.
        """
        if not isinstance(weights_filename, type(None)):
            safe_path = validate_path(
                weights_filename, allowed_extensions=[".pth", ".pt", ".bin"]
            )
            if os.path.isfile(safe_path):
                self.load_state_dict(
                    torch.load(safe_path, weights_only=True, map_location=device)
                )
                self.eval()
                logger.info(
                    "gaussian_3d_volume model weights loaded: {}".format(safe_path)
                )
