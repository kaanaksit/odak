import sys
import torch
from odak.learn.tools.matrix import smooth_pad


def test_smooth_pad_2d_tensor():
    field = torch.ones(10, 10) * 100
    field_padded = smooth_pad(field)
    assert field_padded.shape == (20, 20)
    # Content at center (5:15, 5:15) should be 100
    assert field_padded[5, 5] == 100.0
    # Edges should fade to 0
    assert field_padded[0, 0] < 1.0


def test_smooth_pad_3d_tensor_channels_first():
    field = torch.ones(3, 10, 10) * 100
    field_padded = smooth_pad(field)
    assert field_padded.shape == (3, 20, 20)
    assert field_padded[0, 5, 5] == 100.0
    assert field_padded[0, 0, 0] < 1.0


def test_smooth_pad_4d_tensor():
    field = torch.ones(1, 3, 10, 10) * 100
    field_padded = smooth_pad(field)
    assert field_padded.shape == (3, 20, 20)
    assert field_padded[0, 5, 5] == 100.0
    assert field_padded[0, 0, 0] < 1.0


def test_smooth_pad_3d_tensor_channels_last():
    field = torch.ones(10, 10, 3) * 100
    field_padded = smooth_pad(field)
    assert field_padded.shape == (20, 20, 3)
    assert field_padded[5, 5, 0] == 100.0
    assert field_padded[0, 0, 0] < 1.0


def test_smooth_pad_4d_tensor_channels_last():
    field = torch.ones(1, 10, 10, 3) * 100
    field_padded = smooth_pad(field)
    assert field_padded.shape == (1, 20, 20, 3)
    assert field_padded[0, 5, 5, 0] == 100.0
    assert field_padded[0, 0, 0, 0] < 1.0


def test_smooth_pad_custom_size():
    field = torch.ones(10, 10) * 100
    field_padded = smooth_pad(field, size=[30, 40])
    assert field_padded.shape == (30, 40)
    assert field_padded[10, 15] == 100.0
    assert field_padded[0, 0] < 1.0


def test_smooth_pad_method_left():
    field = torch.ones(10, 10) * 100
    field_padded = smooth_pad(field, method="left")
    assert field_padded.shape == (20, 20)
    # Content is at left (0:10, 0:10), so [0,0] should be 100
    assert field_padded[0, 0] == 100.0
    # Far right should fade to 0
    assert field_padded[19, 19] < 1.0


def test_smooth_pad_method_center():
    field = torch.ones(10, 10) * 100
    field_padded = smooth_pad(field, method="center")
    assert field_padded.shape == (20, 20)
    assert field_padded[5, 5] == 100.0
    assert field_padded[0, 0] < 1.0


def test_smooth_pad_custom_smooth_factor():
    field = torch.ones(10, 10) * 100
    field_padded_soft = smooth_pad(field, smooth_factor=[0.5, 0.5])
    field_padded_hard = smooth_pad(field, smooth_factor=[2.0, 2.0])

    assert field_padded_soft.shape == (20, 20)
    assert field_padded_hard.shape == (20, 20)

    # Both should have content at full value at center
    assert field_padded_soft[5, 5] == 100.0
    assert field_padded_hard[5, 5] == 100.0

    # Both should fade to near zero at edges
    assert field_padded_soft[0, 0] < 1.0
    assert field_padded_hard[0, 0] < 1.0

    # At intermediate positions, hard factor should have smaller values (faster falloff)
    assert field_padded_hard[3, 5].item() < field_padded_soft[3, 5].item()


def test_smooth_pad_device_preservation():
    device = torch.device("cpu")
    field = torch.ones(10, 10, device=device) * 100
    field_padded = smooth_pad(field)
    assert field_padded.device == device


def test_smooth_pad_dtype_preservation():
    field = torch.ones(10, 10, dtype=torch.float32) * 100
    field_padded = smooth_pad(field)
    assert field_padded.dtype == torch.float32


def run_all_tests():
    tests = [
        test_smooth_pad_2d_tensor,
        test_smooth_pad_3d_tensor_channels_first,
        test_smooth_pad_4d_tensor,
        test_smooth_pad_3d_tensor_channels_last,
        test_smooth_pad_4d_tensor_channels_last,
        test_smooth_pad_custom_size,
        test_smooth_pad_method_left,
        test_smooth_pad_method_center,
        test_smooth_pad_custom_smooth_factor,
        test_smooth_pad_device_preservation,
        test_smooth_pad_dtype_preservation,
    ]

    failed = 0
    for test_func in tests:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except AssertionError as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
