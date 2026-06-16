import torch
from odak.learn.wave.optimizers import multi_color_hologram_optimizer
from odak.learn.wave.propagators import propagator


def test_weights_dictionary():
    """Test that weights can be passed as dictionary with keys: image, light, eyebox, phase"""
    wavelengths = [450e-9, 532e-9, 633e-9]
    resolution = [64, 64]
    targets = torch.ones(1, 1, resolution[0], resolution[1])
    
    prop = propagator(
        resolution=resolution,
        wavelengths=wavelengths,
        pixel_pitch=8e-6,
        number_of_frames=3,
        number_of_depth_layers=1,
        volume_depth=0.01,
        image_location_offset=0.005,
        propagation_type="Impulse Response Fresnel",
        device="cpu",
    )
    
    optimizer = multi_color_hologram_optimizer(
        wavelengths=wavelengths,
        resolution=resolution,
        targets=targets,
        propagator=prop,
        number_of_frames=3,
        number_of_depth_layers=1,
        method="multi-color",
        device="cpu",
    )
    
    weights = {
        "image": 1.0,
        "light": 1.0,
        "eyebox": 0.0,
        "phase": 0.0,
    }
    
    result = optimizer.optimize(
        number_of_iterations=1,
        weights=weights,
    )
    
    assert len(result) == 6
    assert result[0].shape[0] == len(wavelengths)


def test_weights_default():
    """Test that default weights work when None is passed"""
    wavelengths = [532e-9]
    resolution = [64, 64]
    targets = torch.ones(1, 1, resolution[0], resolution[1])
    
    prop = propagator(
        resolution=resolution,
        wavelengths=wavelengths,
        pixel_pitch=8e-6,
        number_of_frames=1,
        number_of_depth_layers=1,
        volume_depth=0.01,
        image_location_offset=0.005,
        propagation_type="Impulse Response Fresnel",
        device="cpu",
    )
    
    optimizer = multi_color_hologram_optimizer(
        wavelengths=wavelengths,
        resolution=resolution,
        targets=targets,
        propagator=prop,
        number_of_frames=1,
        number_of_depth_layers=1,
        method="conventional",
        device="cpu",
    )
    
    result = optimizer.optimize(
        number_of_iterations=1,
        weights=None,
    )
    
    assert len(result) == 6


def test_weights_with_eyebox():
    """Test that eyebox weight works when set > 0"""
    wavelengths = [532e-9]
    resolution = [64, 64]
    targets = torch.ones(1, 1, resolution[0], resolution[1])
    
    prop = propagator(
        resolution=resolution,
        wavelengths=wavelengths,
        pixel_pitch=8e-6,
        number_of_frames=1,
        number_of_depth_layers=1,
        volume_depth=0.01,
        image_location_offset=0.005,
        propagation_type="Impulse Response Fresnel",
        device="cpu",
    )
    
    optimizer = multi_color_hologram_optimizer(
        wavelengths=wavelengths,
        resolution=resolution,
        targets=targets,
        propagator=prop,
        number_of_frames=1,
        number_of_depth_layers=1,
        method="conventional",
        device="cpu",
    )
    
    weights = {
        "image": 1.0,
        "light": 0.0,
        "eyebox": 1.0,
        "phase": 0.0,
    }
    
    result = optimizer.optimize(
        number_of_iterations=1,
        weights=weights,
        eyebox={"offset": [0.0, 0.0], "diameter": 50},
    )
    
    assert len(result) == 6


if __name__ == "__main__":
    print("Testing weights dictionary...")
    test_weights_dictionary()
    print("✓ test_weights_dictionary passed")
    
    print("Testing default weights (None)...")
    test_weights_default()
    print("✓ test_weights_default passed")
    
    print("Testing eyebox weights...")
    test_weights_with_eyebox()
    print("✓ test_weights_with_eyebox passed")
    
    print("\nAll tests passed!")
