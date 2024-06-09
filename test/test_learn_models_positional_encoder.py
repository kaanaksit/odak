import torch
import odak
import sys
import random
import os
from tqdm import tqdm


def test(output_directory = 'test_output'):
    odak.tools.check_directory(output_directory)
    device_name = 'cpu'
    filename = './test/data/fruit_lady.png'
    test_filename  = '{}/positional_encoder_multi_layer_perceptron_estimation.png'.format(output_directory)
    weights_filename = '{}/positional_encoder_multi_layer_perceptron_weights.pt'.format(output_directory)
    learning_rate = 1e-4
    no_epochs = 5
    save_at_every = 5000
    number_of_batches = 1
    positional_encoding_level = 24
    dimensions = [2, 256, 256, 256, 3]
    dimensions[0] = dimensions[0] + dimensions[0] * 2 * positional_encoding_level
    device = torch.device(device_name)
    positional_encoder = odak.learn.models.components.positional_encoder(L = positional_encoding_level)
    model = odak.learn.models.multi_layer_perceptron(
                                                     dimensions = dimensions,
                                                     activation = torch.nn.ReLU(),
                                                     bias = False,
                                                     model_type = 'conventional'
                                                    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    image = odak.learn.tools.load_image(filename, normalizeby = 255., torch_style = False)[:, :, 0:3].to(device)
    batches = get_batches(image, number_of_batches).to(device)
    loss_function = torch.nn.MSELoss()
    epochs = tqdm(range(no_epochs), leave = False, dynamic_ncols = True)    
    if os.path.isfile(weights_filename):
        model.load_state_dict(torch.load(weights_filename))
        model.eval()
        print('Model weights loaded: {}'.format(weights_filename))
    try:
        for epoch_id in epochs:
            test_loss, estimation = trial(image, batches, loss_function, model, positional_encoder)
            train_loss = train(image, batches, optimizer, loss_function, model, positional_encoder)
            description = 'train loss: {:.5f}, test loss:{:.5f}'.format(train_loss, test_loss)
            epochs.set_description(description)
            if epoch_id % save_at_every == 0:
                odak.learn.tools.save_image(test_filename, estimation, cmin = 0., cmax = 1.)
        torch.save(model.state_dict(), weights_filename)
        print('Model weights save: {}'.format(weights_filename))
        odak.learn.tools.save_image(test_filename, estimation, cmin = 0., cmax = 1.)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), weights_filename)
        print('Model weights save: {}'.format(weights_filename))
        odak.learn.tools.save_image(test_filename, estimation, cmin = 0., cmax = 1.)
        assert True == True
    assert True == True


def get_batches(image, number_of_batches = 100):
    xs = torch.arange(image.shape[0])
    ys = torch.arange(image.shape[1])
    XS, YS = torch.meshgrid(xs, ys, indexing = 'ij')
    XS = XS.reshape(number_of_batches, -1, 1)
    YS = YS.reshape(number_of_batches, -1, 1)
    batches = torch.concat((XS, YS), axis = 2).float()
    return batches


def train(output_values, input_values, optimizer, loss_function, model, positional_encoder):
    total_loss = 0.
    for input_value in input_values:
        optimizer.zero_grad()
        normalized_input_value = torch.zeros_like(input_value)
        normalized_input_value[:, 0] = input_value[:, 0] / output_values.shape[0]
        normalized_input_value[:, 1] = input_value[:, 1] / output_values.shape[1]
        input_value = positional_encoder(input_value)
        estimation = model(input_value)
        ground_truth = output_values[input_value[:, 0].int(), input_value[:, 1].int(), :]
        loss = loss_function(estimation, ground_truth)
        loss.backward(retain_graph = True)
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def trial(output_values, input_values, loss_function, model, positional_encoder):
    estimated_image = torch.zeros_like(output_values)
    for input_value in input_values:
        torch.no_grad()
        normalized_input_value = torch.zeros_like(input_value)
        normalized_input_value[:, 0] = input_value[:, 0] / output_values.shape[0]
        normalized_input_value[:, 1] = input_value[:, 1] / output_values.shape[1]
        input_value = positional_encoder(input_value)
        estimation = model(input_value)
        ground_truth = output_values[input_value[:, 0].int(), input_value[:, 1].int(), :]
        estimated_image[input_value[:, 0].int(), input_value[:, 1].int(), :] = estimation
        loss = loss_function(estimation, ground_truth)
    loss = loss_function(estimated_image, output_values)
    return loss, estimated_image


if  __name__ ==  '__main__':
    sys.exit(test())
