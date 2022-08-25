import torch


def check_loss_inputs(loss_name, image, target):
        if image.size() != target.size():
            raise Exception(
                loss_name + " ERROR: Input and target must have same dimensions.")
        if len(image.size()) != 4:
            raise Exception(
                loss_name + " ERROR: Input and target must have 4 dimensions in total (NCHW format).")
        if image.size(1) != 1 and image.size(1) != 3:
            raise Exception(
                loss_name + """ ERROR: Inputs should have either 1 or 3 channels 
                (1 channel for grayscale, 3 for RGB or YCrCb).
                Ensure inputs have 1 or 3 channels and are in NCHW format.""")
