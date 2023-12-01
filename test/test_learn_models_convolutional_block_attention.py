import torch
import sys
import odak


def test():
    x = torch.randn(1, 32, 256, 256)
    cbam_inference = odak.learn.models.convolutional_block_attention(gate_channels = 32)
    y = cbam_inference(x)
    assert x.shape == y.shape, "CBAM output shape does not match input shape"


if __name__ == '__main__':
    sys.exit(test())
