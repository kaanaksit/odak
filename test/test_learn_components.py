import odak.learn.models.components as components
import torch
import sys


def test():
    # test residual block
    x = torch.randn(1, 2, 32, 32)
    residual_inference = components.residual_layer()
    residual_inference(x)
    # test convolution layer
    convolution_inference = components.convolution_layer()
    convolution_inference(x)
    # test double convolution layer
    double_convolution_inference = components.double_convolution()
    double_convolution_inference(x)
    # test normalization layer
    normalization_inference = components.normalization()
    normalization_inference(x)
    # test attention layer
    residual_attention_layer_inference = components.residual_attention_layer()
    residual_attention_layer_inference(x, x)
    # test self-attention layer
    non_local_layer_inference = components.non_local_layer(
        input_channels=2, bottleneck_channels=1
    )
    non_local_layer_inference(x)
    assert True


if __name__ == "__main__":
    sys.exit(test())
