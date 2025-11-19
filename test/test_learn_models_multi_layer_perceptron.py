import torch
import odak
import sys
import random
import os
from tqdm import tqdm


def test(
         output_directory = 'test_output',
         header = 'test/test_learn_models_multi_layer_perceptron.py',
        ):
    odak.tools.check_directory(output_directory)
    filename = './test/data/fruit_lady.png'
    test_filename  = '{}/multi_layer_perceptron_estimation.png'.format(output_directory)
    weights_filename = '{}/multi_layer_perceptron_model_weights.pt'.format(output_directory)
    learning_rate = 1e-4
    no_epochs = 2 #100000
    dimensions = [2, 64, 64, 64, 64, 64, 64, 3]
    device_name = 'cpu'
    save_at_every = 1000
    model_type = 'FILM SIREN'
    test_resolution_scale = 4
    inject_noise = True
    noise_ratio = 1e-3
    device = torch.device(device_name)
    model = odak.learn.models.multi_layer_perceptron(
                                                     dimensions = dimensions,
                                                     activation = torch.nn.Tanh(),
                                                     bias = True,
                                                     model_type = model_type,
                                                     input_multiplier = 100,
                                                     siren_multiplier = 1.
                                                    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    scheduler =  torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters = no_epochs, power = 1.)
    image = odak.learn.tools.load_image(filename, normalizeby = 255., torch_style = False)[:, :, 0:3].to(device)
    original_resolution = image.shape
    image = image.reshape(-1, original_resolution[-1])
    test_resolution = [original_resolution[0] * test_resolution_scale, original_resolution[1] * test_resolution_scale, 3]
    train_batches = get_batches(original_resolution).to(device)
    test_batches = get_batches(test_resolution).to(device)
    loss_function = torch.nn.MSELoss(reduction = 'mean')
    epochs = tqdm(range(no_epochs), leave = False, dynamic_ncols = True)    
    if os.path.isfile(weights_filename):
        model.load_state_dict(torch.load(weights_filename))
        model.eval()
        odak.log.logger.info('{} -> Model weights loaded: {}'.format(header, weights_filename))
    try:
        for epoch_id in epochs:
            train_loss = train(
                               output_values = image, 
                               input_values = train_batches,
                               optimizer = optimizer, 
                               scheduler = scheduler,
                               loss_function = loss_function,
                               model = model,
                               inject_noise = inject_noise,
                               noise_ratio = noise_ratio
                              )
            current_learning_rate = optimizer.param_groups[0]['lr']
            description = 'train loss: {:.8f}, learning rate: {:.8f}'.format(train_loss, current_learning_rate)
            epochs.set_description(description)
            if epoch_id % save_at_every == 0: 
                estimation = trial(test_batches, model, test_resolution)
                odak.learn.tools.save_image(test_filename, estimation, cmin = 0., cmax = 1.)
        estimation = trial(test_batches, model, test_resolution)
        odak.learn.tools.save_image(test_filename, estimation, cmin = 0., cmax = 1.)
        odak.log.logger.info('{} -> {}'.format(header, description))
        torch.save(model.state_dict(), weights_filename)
        odak.log.logger.info('{} -> Model weights save: {}'.format(header, weights_filename))
    except KeyboardInterrupt:
        odak.log.logger.info('{} -> {}'.format(header, description))
        torch.save(model.state_dict(), weights_filename)
        odak.log.logger.info('{} -> Model weights save: {}'.format(header, weights_filename))
        odak.learn.tools.save_image(test_filename, estimation, cmin = 0., cmax = 1.)
        assert True == True
    assert True == True


def get_batches(size):
    xs = torch.linspace(-1, 1, steps = size[0])
    ys = torch.linspace(-1, 1, steps = size[1])
    XS, YS = torch.meshgrid(xs, ys, indexing = 'ij')
    batches = torch.concat((XS.reshape(-1, 1), YS.reshape(-1, 1)), axis = 1).float()
    return batches


def train(
          output_values,
          input_values,
          optimizer,
          scheduler,
          loss_function,
          model,
          inject_noise = False,
          noise_ratio = 1e-3
         ):
    optimizer.zero_grad()
    estimation = model(input_values)
    gt = output_values.detach().clone()
    if inject_noise:
        gt += (gt.max() - gt.min()) * noise_ratio * torch.randn_like(gt)
    loss = loss_function(estimation, gt)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.item())


def trial(input_values, model, resolution):
    torch.no_grad()
    estimation = model(input_values)
    estimated_image = estimation.reshape(resolution)
    return estimated_image


if  __name__ ==  '__main__':
    sys.exit(test())
