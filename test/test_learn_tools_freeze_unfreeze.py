import torch
import sys
import odak


header = './test_learn_tools_freeze_unfreeze.py'


def check_parameters_gradient(model):
    odak.log.logger.info('{} -> {}'.format(header, model.parameters))
    for (fullname, parameter) in model.named_parameters():
        odak.log.logger.info('{} -> Name: {}, Requires grad: {}'.format(header, fullname, parameter.requires_grad))


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
