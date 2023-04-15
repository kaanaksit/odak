import odak.learn.models.components as components
import torch
import sys

def test():
    # test residual block
    x = torch.randn(1, 2, 32, 32)
    residual_inference= components.residual_layer()
    y = residual_inference(x)
    print(y.shape)
    # test convolution layer
    convolution_inference = components.convolution_layer()
    y = convolution_inference(x)
    print(y.shape)
    # test double convolution layer
    double_convolution_inference = components.double_convolution()
    y = double_convolution_inference(x)
    print(y.shape)
    # test normalization layer
    normalization_inference = components.normalization()
    y = normalization_inference(x)
    print(y.shape)
    # test attention layer
    residual_attention_layer_inference = components.residual_attention_layer()
    y = residual_attention_layer_inference(x , x)
    print(y.shape)
    # test self-attention layer
    non_local_layer_inference = components.non_local_layer(input_channels=2,
                                                                      bottleneck_channels=1
                                                                      )
    z = non_local_layer_inference(x)
    print(z.shape)
    assert True == True

if __name__ == '__main__':
    sys.exit(test())