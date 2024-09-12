import torch
import odak
import sys
import numpy as np
import matplotlib.pyplot as plt


def train(output_values, input_values, optimizer, loss_function, model, channel):
    model.train()
    optimizer.zero_grad()
    estimation = model(input_values)
    loss = loss_function(estimation, output_values[:, channel])
    loss.backward(retain_graph = True)
    optimizer.step()
    return loss.item()


def trial(input_values, loss_function, model):
    model.eval()
    with torch.no_grad():
        estimation = model(input_values)
    return estimation


def run_training(
                 wavelengths, 
                 rgb_values, 
                 learning_rate = 1e-4, 
                 no_epochs = 2000, 
                 dimensions = [1, 64, 64, 1],
                 model_type = 'conventional',
                 activation = torch.nn.ReLU(),
                 bias = True,
                 input_multiplier = 1.,
                 device = torch.device('cpu')
                ):
    models = []
    optimizers = []
    for channel_index in range(3):  # Loop to create three separate models for R, G, B
        model = odak.learn.models.multi_layer_perceptron(
                                                        dimensions = dimensions, 
                                                        activation = activation,
                                                        bias = bias, 
                                                        model_type = model_type,
                                                        input_multiplier = input_multiplier
                                                        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
        loss_function = torch.nn.MSELoss()
        for epoch in range(no_epochs):
            loss = train(rgb_values, wavelengths, optimizer, loss_function, model, channel_index)
            if epoch % 100 == 0:
                print(f'Channel {channel_index} - Epoch {epoch + 1}, Loss: {loss}')
        models.append(model)
        optimizers.append(optimizer)
    return models

def plot_predictions(models, wavelengths, rgb_values, device):
    new_wavelengths = torch.linspace(300, 700, 401, device=device).reshape(-1, 1)
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'blue']
    labels = ['Red', 'Green', 'Blue']
    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            predicted_rgb_scaled = model(new_wavelengths)
        plt.plot(
                new_wavelengths.detach().cpu().numpy(), 
                predicted_rgb_scaled.detach().cpu().numpy(), 
                label=f'Predicted {labels[i]}', color=colors[i]
                )
        plt.scatter(
                    wavelengths.detach().cpu().numpy(), 
                    rgb_values[:, i].detach().cpu().numpy(), 
                    color=colors[i], 
                    label=f'{labels[i]} Training', 
                    alpha=0.5
                    )
    plt.title('Predicted RGB Intensity vs. Wavelength')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()



def test(device = torch.device('cuda')):
    wavelengths_OLED = torch.tensor([300, 310, 311, 360, 378, 380, 385, 390, 395, 400, 410, 424, 425, 426, 427, 440,
        450, 470, 480, 490, 500, 505, 509, 510, 511, 539, 540, 541, 560, 590,
        600, 650, 700], device='cuda:0').unsqueeze(-1)
    rgb_values_OLED = torch.tensor([
        [0.0000, 0.0000, 0.0100],
        [0.0000, 0.0000, 0.0200],
        [0.0000, 0.0000, 0.0210],
        [0.0000, 0.0000, 0.1500],
        [0.0000, 0.0000, 0.2100],
        [0.0000, 0.0000, 0.2500],
        [0.0000, 0.0000, 0.5000],
        [0.0000, 0.0000, 0.5200],
        [0.0000, 0.0000, 0.5300],
        [0.0000, 0.0000, 0.5500],
        [0.0000, 0.0000, 0.5800],
        [0.0000, 0.0000, 0.9900],
        [0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 0.9900],
        [0.0000, 0.0000, 0.9800],
        [0.0000, 0.0000, 0.5500],
        [0.0000, 0.0000, 0.4500],
        [0.0000, 0.0000, 0.4500],
        [0.0000, 0.0100, 0.1500],
        [0.0000, 0.0100, 0.1800],
        [0.0000, 0.1200, 0.0500],
        [0.0000, 0.6000, 0.0300],
        [0.0000, 0.8900, 0.0500],
        [0.0000, 0.9000, 0.0500],
        [0.0000, 0.8900, 0.0500],
        [0.0000, 0.0000, 0.4600],
        [0.0500, 0.6500, 0.0100],
        [0.0000, 0.0000, 0.4600],
        [0.3000, 0.8500, 0.0100],
        [1.0000, 0.3000, 0.0000],
        [0.6000, 0.3000, 0.0000],
        [0.1000, 0.0900, 0.0000],
        [0.0100, 0.0200, 0.0000]], device='cuda:0').unsqueeze(-1)
    # wavelengths_OLED = torch.linspace(300., 700., 10).to(device).unsqueeze(-1)
    # rgb_values_OLED = torch.linspace(-1000., 1000., 10).to(device).unsqueeze(-1)
    # rgb_values_OLED = rgb_values_OLED.repeat(1, 3).unsqueeze(-1)
    model = run_training(
                         wavelengths_OLED, 
                         rgb_values_OLED, 
                         learning_rate = 1e-3, 
                         no_epochs = 8000, 
                         dimensions = [1, 64, 64, 1],
                         input_multiplier = 1.,
                         model_type = 'conventional',
                         bias = True,
                         activation = torch.nn.Tanh(),
                         device = device
                        )
    plot_predictions(model, wavelengths_OLED, rgb_values_OLED, device)
    

    # wavelengths_LED = torch.tensor([300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 650, 521, 450], device = device).unsqueeze(-1)
    # rgb_values_LED = torch.tensor(
    # [
    #     [3.31e-134, 1.78e-95, 7.73e-17],
    #     [1.93e-119, 3.48e-79, 6.02e-13],
    #     [1.53e-105, 1.94e-64, 1.30e-09],
    #     [1.63e-92, 3.09e-51, 7.84e-07],
    #     [2.36e-80, 1.41e-39, 0.000131],
    #     [4.62e-69, 1.83e-29, 0.006105],
    #     [1.22e-58, 6.81e-21, 0.079474],
    #     [4.38e-49, 7.23e-14, 0.583601],
    #     [2.13e-40, 2.19e-08, 0.583601],
    #     [1.40e-32, 0.000190, 0.079474],
    #     [1.24e-25, 0.046990, 0.006105],
    #     [1.49e-19, 0.983870, 0.000131],
    #     [2.43e-14, 0.067219, 7.84e-07],
    #     [5.35e-10, 0.000386, 1.30e-09],
    #     [1.60e-06, 6.37e-08, 6.02e-13],
    #     [0.000643, 2.99e-13, 7.73e-17],
    #     [0.035142, 4.03e-20, 2.76e-21],
    #     [0.450603, 1.55e-28, 2.74e-26],
    #     [0.450603, 1.70e-38, 7.57e-32],
    #     [0.035142, 5.32e-50, 5.81e-38],
    #     [0.000643, 4.76e-63, 1.24e-44],
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]

    # ]
    # , device = device).unsqueeze(-1)
    # model = run_training(
    #                      wavelengths_LED, 
    #                      rgb_values_LED, 
    #                      learning_rate = 5e-4, 
    #                      no_epochs = 150000, 
    #                      dimensions = [1, 64, 64, 1],
    #                      input_multiplier = 1.,
    #                      model_type = 'conventional',
    #                      bias = True,
    #                      activation = torch.nn.Tanh(),
    #                      device = device
    #                     )
    # plot_predictions(model, wavelengths_LED, rgb_values_LED, device)

if __name__ == '__main__':
   sys.exit(test()) 