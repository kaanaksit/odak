import sys
import odak
import torch

def test():
    try:
        import pycvvdp
    except ImportError:
        print('ColorVideoVDP is missing, consider installing by visiting: https://github.com/gfxdisp/ColorVideoVDP')
        assert True == True
    if 'pycvvdp' in sys.modules:
        import pycvvdp
        print('ColorVideoVDP is imported.')
        main(pycvvdp)
    assert True == True


def main(
         pycvvdp,
         output_directory = 'test_output', 
         device = torch.device('cpu'),
         pixels_per_degree = 60,
         display_standard = 'standard_4k',
         display_config_paths = [],
         display_temp_padding = 'replicate',
        ):
    odak.tools.check_directory(output_directory)
    image = odak.learn.tools.load_image(
                                        './test/data/fruit_lady.png',
                                        normalizeby = 255.,
                                        torch_style = True
                                       ).to(device)
    image_noisy = image * 0.9 + torch.rand_like(image) * 0.1

    display_geometry = pycvvdp.vvdp_display_geometry(
                                                     [image.shape[-2], image.shape[-1]],
                                                     ppd = pixels_per_degree 
                                                    )
    display_photometry = pycvvdp.vvdp_display_photometry.load(
                                                              display_standard,
                                                              config_paths = display_config_paths
                                                             )
    colorvdp = pycvvdp.cvvdp(
                             display_photometry = display_photometry,
                             display_geometry = display_geometry,
                             temp_padding = display_temp_padding,
                             device = device
                            )
    loss = colorvdp.loss(image_noisy, image, dim_order = 'CHW')
    print(loss)
    assert True == True


if __name__ == '__main__':
    sys.exit(test())








