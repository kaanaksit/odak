import torch
import sys
import odak.learn.models.components as components

def test():
    # test cbam layer
    x = torch.randn(1, 32, 256, 256)
    # Instantiate the CBAM layer with 2 input channels
    cbam_inference = components.CBAM(gate_channels=32)
    y = cbam_inference(x)
    print(y.shape)
    # Basic check to confirm output tensor has the same shape as input
    assert x.shape == y.shape, "CBAM output shape does not match input shape"

if __name__ == '__main__':
    sys.exit(test())
