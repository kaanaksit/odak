import torch
import odak
import sys
import random
import os
from tqdm import tqdm


def test(output_directory = 'test_output'):
    odak.tools.check_directory(output_directory)
    filename = './test/data/fruit_lady.png'
    test_filename  = '{}/multi_layer_perceptron_estimation.png'.format(output_directory)
    weights_filename = '{}/multi_layer_perceptron_model_weights.pt'.format(output_directory)
    learning_rate = 1e-4
    no_epochs = 5
    dimensions = [2, 128, 128, 3]
    device_name = 'cpu'
    save_at_every = 2000
    model_type = 'FILM SIREN'
    test_resolution_scale = 4
    device = torch.device(device_name)
    model = odak.learn.models.multi_layer_perceptron(
                                                     dimensions = dimensions,
                                                     activation = torch.nn.Tanh(),
                                                     bias = True,
                                                     model_type = model_type,
                                                     input_multiplier = 200,
                                                    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    image = odak.learn.tools.load_image(filename, normalizeby = 255., torch_style = False)[:, :, 0:3].to(device)
    original_resolution = image.shape
    image = (image * 2.) - 1.
    image = image.reshape(-1, original_resolution[-1])
    test_resolution = [original_resolution[0] * test_resolution_scale, original_resolution[1] * test_resolution_scale, 3]
    train_batches = get_batches(original_resolution).to(device)
    test_batches = get_batches(test_resolution).to(device)
    loss_function = torch.nn.MSELoss(reduction = 'sum')
    epochs = tqdm(range(no_epochs), leave = False, dynamic_ncols = True)    
    if os.path.isfile(weights_filename):
        model.load_state_dict(torch.load(weights_filename))
        model.eval()
        print('Model weights loaded: {}'.format(weights_filename))
    try:
        for epoch_id in epochs:
            train_loss = train(image, train_batches, optimizer, loss_function, model)
            description = 'train loss: {:.5f}'.format(train_loss)
            epochs.set_description(description)
            if epoch_id % save_at_every == 0: 
                estimation = trial(test_batches, model, test_resolution)
                odak.learn.tools.save_image(test_filename, estimation, cmin = 0., cmax = 1.)
        estimation = trial(test_batches, model, test_resolution)
        odak.learn.tools.save_image(test_filename, estimation, cmin = 0., cmax = 1.)
        print(description)
        torch.save(model.state_dict(), weights_filename)
        print('Model weights save: {}'.format(weights_filename))
    except KeyboardInterrupt:
        print(description)
        torch.save(model.state_dict(), weights_filename)
        print('Model weights save: {}'.format(weights_filename))
        odak.learn.tools.save_image(test_filename, estimation, cmin = 0., cmax = 1.)
        assert True == True
    assert True == True


def get_batches(size):
    xs = torch.linspace(-1, 1, steps = size[0])
    ys = torch.linspace(-1, 1, steps = size[1])
    XS, YS = torch.meshgrid(xs, ys, indexing = 'ij')
    batches = torch.concat((XS.reshape(-1, 1), YS.reshape(-1, 1)), axis = 1).float()
    return batches


def train(output_values, input_values, optimizer, loss_function, model):
    optimizer.zero_grad()
    estimation = model(input_values)
    loss = loss_function(estimation, output_values)
    loss.backward()
    optimizer.step()
    return loss.item()


def trial(input_values, model, resolution):
    torch.no_grad()
    estimation = model(input_values)
    estimated_image = (estimation.reshape(resolution) + 1.) / 2.
    return estimated_image


if  __name__ ==  '__main__':
    sys.exit(test())
