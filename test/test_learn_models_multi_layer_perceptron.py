import torch
import odak
import sys
import random
import os
from tqdm import tqdm


def main():
    filename = './test/fruit_lady.png'
    test_filename  = './estimation.png'
    weights_filename = 'model_weights.pt'
    learning_rate = 1e-3
    no_epochs = 20
    number_of_batches = 1
    dimensions = [2, 128, 128, 3]
    device_name = 'cpu'
    save_at_every = 500
    device = torch.device(device_name)
    model = odak.learn.models.multi_layer_perceptron(
                                                     dimensions = dimensions,
                                                     activation = torch.nn.Tanh(),
                                                     bias = False,
                                                     model_type = 'FILM SIREN'
                                                    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    image = odak.learn.tools.load_image(filename, normalizeby = 255., torch_style = False)[:, :, 0:3].to(device)
    image = odak.learn.tools.zero_pad(image)
    train_batches = get_train_batches(image, number_of_batches).to(device)
    test_batches = get_test_batches(image, number_of_batches).to(device)
    loss_function = torch.nn.MSELoss()
    epochs = tqdm(range(no_epochs), leave = False, dynamic_ncols = True)    
    if os.path.isfile(weights_filename):
        model.load_state_dict(torch.load(weights_filename))
        model.eval()
        print('Model weights loaded: {}'.format(weights_filename))
    try:
        for epoch_id in epochs:
            test_loss, estimation = trial(image, test_batches, loss_function, model)
            train_loss = train(image, train_batches, optimizer, loss_function, model)
            description = 'train loss: {:.5f}, test loss:{:.5f}'.format(train_loss, test_loss)
            epochs.set_description(description)
            if epoch_id % save_at_every == 0: 
                odak.learn.tools.save_image(test_filename, estimation, cmin = 0., cmax = 1.)
        print(description)
        torch.save(model.state_dict(), weights_filename)
        print('Model weights save: {}'.format(weights_filename))
        odak.learn.tools.save_image(test_filename, estimation, cmin = 0., cmax = 1.)
    except KeyboardInterrupt:
        print(description)
        torch.save(model.state_dict(), weights_filename)
        print('Model weights save: {}'.format(weights_filename))
        odak.learn.tools.save_image(test_filename, estimation, cmin = 0., cmax = 1.)
        assert True == True
    assert True == True


def get_test_batches(image, number_of_batches = 100):
    xs = torch.arange(image.shape[0])
    ys = torch.arange(image.shape[1])
    XS, YS = torch.meshgrid(xs, ys, indexing = 'ij')
    XS = XS.reshape(number_of_batches, -1, 1)
    YS = YS.reshape(number_of_batches, -1, 1)
    batches = torch.concat((XS, YS), axis = 2).float()
    return batches


def get_train_batches(image, number_of_batches = 100):
    xs = torch.arange(image.shape[0] // 2) + image.shape[0] // 4
    ys = torch.arange(image.shape[1] // 2) + image.shape[1] // 4
    XS, YS = torch.meshgrid(xs, ys, indexing = 'ij')
    XS = XS.reshape(number_of_batches, -1, 1)
    YS = YS.reshape(number_of_batches, -1, 1)
    batches = torch.concat((XS, YS), axis = 2).float()
    return batches


def get_test_batches(image, number_of_batches = 100):
    xs = torch.arange(image.shape[0])
    ys = torch.arange(image.shape[1])
    XS, YS = torch.meshgrid(xs, ys, indexing = 'ij')
    XS = XS.reshape(number_of_batches, -1, 1)
    YS = YS.reshape(number_of_batches, -1, 1)
    batches = torch.concat((XS, YS), axis = 2).float()
    return batches



def train(output_values, input_values, optimizer, loss_function, model):
    total_loss = 0.
    for input_value in input_values:
        optimizer.zero_grad()
        estimation = model(input_value)
        ground_truth = output_values[input_value[:, 0].int(), input_value[:, 1].int(), :]
        loss = loss_function(estimation, ground_truth)
        loss.backward(retain_graph = True)
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def trial(output_values, input_values, loss_function, model):
    estimated_image = torch.zeros_like(output_values)
    for input_value in input_values:
        torch.no_grad()
        estimation = model(input_value)
        ground_truth = output_values[input_value[:, 0].int(), input_value[:, 1].int(), :]
        estimated_image[input_value[:, 0].int(), input_value[:, 1].int(), :] = estimation
        loss = loss_function(estimation, ground_truth)
    estimated_cropped_image = odak.learn.tools.crop_center(estimated_image)
    output_values = odak.learn.tools.crop_center(output_values)
    loss = loss_function(estimated_cropped_image, output_values)
    return loss, estimated_image


if  __name__ ==  '__main__':
    sys.exit(main())
