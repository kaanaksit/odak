# Spatial Steerable Pyramid

`odak.learn.perception.SpatialSteerablePyramid()`

This implements a real-valued steerable pyramid where the filtering is carried out spatially (using convolution) as opposed to multiplication in the Fourier domain. This has a number of optimisations over previous implementations that increase efficiency, but introduce some reconstruction error.

**Parameters:**

    use_bilinear_downup     : bool
                                This uses bilinear filtering when upsampling/downsampling, rather than the original approach
                                of applying a large lowpass kernel and sampling even rows/columns
    n_channels              : int
                                Number of channels in the input images (e.g. 3 for RGB input)
    filter_size             : int
                                Desired size of filters (e.g. 3 will use 3x3 filters).
    n_orientations          : int
                                Number of oriented bands in each level of the pyramid.
    filter_type             : str
                                This can be used to select smaller filters than the original ones if desired.
                                full: Original filter sizes
                                cropped: Some filters are cut back in size by extracting the centre and scaling as appropriate.
                                trained: Same as reduced, but the oriented kernels are replaced by learned 5x5 kernels.
    device                  : torch.device
                                torch device the input images will be supplied from.

## Methods

`construct_pyramid()`

Constructs and returns a steerable pyramid for the provided image.

**Parameters**

        image               : torch.tensor
                                The input image, in NCHW format. The number of channels C should match num_channels
                                when the pyramid maker was created.
        n_levels            : int
                                Number of levels in the constructed steerable pyramid.
        multiple_highpass   : bool
                                If true, computes a highpass for each level of the pyramid.
                                These extra levels are redundant (not used for reconstruction). 
        
**Returns**

        pyramid             : list of dicts of torch.tensor
                                The computed steerable pyramid.
                                Each level is an entry in a list. The pyramid is ordered from largest levels to smallest levels.
                                Each level is stored as a dict, with the following keys:
                                "h" Highpass residual
                                "l" Lowpass residual
                                "b" Oriented bands (a list of torch.tensor)

`reconstruct-from_pyramid()`

Reconstructs an input image from a steerable pyramid

**Parameters**

        pyramid : list of dicts of torch.tensor
                    The steerable pyramid.
                    Should be in the same format as output by construct_steerable_pyramid().
                    The number of channels should match num_channels when the pyramid maker was created.

**Returns**

        image   : torch.tensor
                    The reconstructed image, in NCHW format.  

## See also

[`Perception`](../../../perception.md)
