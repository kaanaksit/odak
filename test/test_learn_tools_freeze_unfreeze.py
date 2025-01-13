import torch
import sys
import odak


def check_parameters_gradient(model):
    print(model.parameters)
    for (fullname, parameter) in model.named_parameters():
        print('Name: {}, Requires grad: {}'.format(fullname, parameter.requires_grad))


def test(
         device = torch.device('cpu'),
        ):
    model = odak.learn.models.multi_layer_perceptron(dimensions = [1, 5, 5])
    odak.learn.tools.freeze(model)
    check_parameters_gradient(model)
    odak.learn.tools.unfreeze(model)
    check_parameters_gradient(model)
    assert True == True

if __name__ == '__main__':
    sys.exit(test())
