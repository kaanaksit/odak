import sys
import odak
import torch
from tqdm import tqdm


def test(
    output_directory="test_output",
    visualization=False,
    device=torch.device("cpu"),
    header="test_learn_tools_grid_sample.py",
):
    odak.tools.check_directory(output_directory)
    
    # Test basic functionality with default parameters
    samples, rotx, roty, rotz = odak.learn.tools.grid_sample()
    
    # Verify the output shapes
    assert samples.shape == torch.Size([100, 3]), "Samples shape should be [100, 3]"
    assert rotx.shape == torch.Size([1, 3, 3]), "Rotation matrix X shape should be [1, 3, 3]"
    assert roty.shape == torch.Size([1, 3, 3]), "Rotation matrix Y shape should be [1, 3, 3]"
    assert rotz.shape == torch.Size([1, 3, 3]), "Rotation matrix Z shape should be [1, 3, 3]"
    
    # Test with custom parameters
    custom_no = [5, 5]
    custom_size = [50.0, 50.0]
    custom_center = [10.0, 10.0, 10.0]
    custom_angles = [45.0, 30.0, 15.0]
    
    samples, rotx, roty, rotz = odak.learn.tools.grid_sample(
        no=custom_no,
        size=custom_size,
        center=custom_center,
        angles=custom_angles
    )
    
    # Verify the output shapes with custom parameters
    assert samples.shape == torch.Size([25, 3]), "Samples shape should be [25, 3] with 5x5 grid"
    assert rotx.shape == torch.Size([1, 3, 3]), "Rotation matrix X shape should be [1, 3, 3]"
    assert roty.shape == torch.Size([1, 3, 3]), "Rotation matrix Y shape should be [1, 3, 3]"
    assert rotz.shape == torch.Size([1, 3, 3]), "Rotation matrix Z shape should be [1, 3, 3]"
    
    # Test with zero angles and simple grid to check values
    samples_zero, _, _, _ = odak.learn.tools.grid_sample(
        no=[3, 3],
        size=[10.0, 10.0],
        center=[0.0, 0.0, 0.0],
        angles=[0.0, 0.0, 0.0]
    )
    
    # Verify that we get correct number of samples
    assert samples_zero.shape == torch.Size([9, 3]), "Should have 9 samples for 3x3 grid"
    
    # Expected positions should be a grid from -5 to 5 in both directions
    expected_positions = torch.tensor([
        [-5.0, -5.0, 0.0],
        [-5.0, 0.0, 0.0],
        [-5.0, 5.0, 0.0],
        [0.0, -5.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [5.0, -5.0, 0.0],
        [5.0, 0.0, 0.0],
        [5.0, 5.0, 0.0],
    ], dtype=torch.float32)
    
    # Check that all positions match expected values (allowing for some numerical precision)
    assert torch.allclose(samples_zero, expected_positions, atol=1e-6), "Grid positions should match expected values"
    
    # Test with non-zero angles to make sure it doesn't crash
    samples_rot, _, _, _ = odak.learn.tools.grid_sample(
        no=[2, 2],
        size=[4.0, 4.0],
        center=[0.0, 0.0, 0.0],
        angles=[90.0, 0.0, 0.0]
    )
    
    assert samples_rot.shape == torch.Size([4, 3]), "Should have 4 samples for 2x2 grid with rotation"
    
    odak.log.logger.info("{} -> Grid sample test passed".format(header))
    
    return True


if __name__ == "__main__":
    sys.exit(test())