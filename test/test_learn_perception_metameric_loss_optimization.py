import sys
import odak
import torch
from tqdm import tqdm


def test(
         filename = 'test/data/sample_rgb.png',
         learning_rate = 1e-2,
         gaze = [0.5, 0.5],
         number_of_steps = 1, # Change this to 150 to run the optimization properly.
         output_directory = 'test_output',
         device = torch.device('cpu'),
         header = 'test/test_learn_perception_metameric_loss_optimization.py'
        ):
    ground_truth = odak.learn.tools.load_image(
                                               filename,
                                               normalizeby = 255.,
                                               torch_style = True
                                               ).to(device)
    ground_truth = ground_truth.unsqueeze(0)
    estimate = torch.rand_like(ground_truth, requires_grad = True)
    metameric_loss = odak.learn.perception.MetamerMSELoss().to(device)
    optimizer = torch.optim.Adam([estimate], lr = learning_rate)
    t = tqdm(range(number_of_steps), leave = False, dynamic_ncols = True)
    for step in t:
        optimizer.zero_grad()
        loss = metameric_loss(estimate, ground_truth, gaze = gaze)
        loss.backward(retain_graph = True)
        optimizer.step()
        description = '{} -> Loss: {}'.format(header, loss.item())
        t.set_description(description)
        if step % 10 == 0:
            odak.tools.check_directory(output_directory)
            odak.learn.tools.save_image(
                                        '{}/metameric_foveation.png'.format(output_directory),
                                        estimate,
                                        cmin = 0.,
                                        cmax = 1.
                                       )
    odak.log.logger.info(description)
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
