import argparse
import logging
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from utils import parse, tensor2img, dict2str
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
from data.HR_dataset import HRDataset
import model as Model
import data as Data
from model.model import DDPM
def train(opt):
    # Initialize logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(dict2str(opt))

    # Initialize dataset and dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = HRDataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = HRDataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    # Initialize the diffusion model
    diffusion = DDPM(opt)
    logger.info('Model [{:s}] is created.'.format(diffusion.__class__.__name__))
    logger.info('Initial Model Finished')

    # Training loop
    for epoch in range(opt['train']['n_epochs']):
        for iteration, train_data in enumerate(train_loader):
            # Forward pass
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            current_step = epoch * len(train_loader) + iteration

        # Validation and model saving
        if epoch % opt['train']['val_freq'] == 0 and epoch != 0:
            # Log the training loss and other relevant information
            logger.info(f'Epoch [{epoch}/{opt["train"]["n_epochs"]}]\t'
                        f'Loss: {diffusion.get_current_loss()}')
            # Set the noise schedule
            diffusion.set_new_noise_schedule()
            validate(opt, diffusion, val_loader, current_step, logger)
            diffusion.set_new_noise_schedule()
            # Save checkpoints
            logger.info('Saving models and training states.')
            diffusion.save_network(epoch)

    logger.info('Training finished.')


def validate(opt, model, val_loader, current_step, logger):

    psnr_sum = 0.0
    ssim_sum = 0.0
    num_samples = 0

    for idx, val_data in enumerate(val_loader):
        model.feed_data(val_data)
        model.test(continous=False)  # You may need to adjust the test function based on your model
        visuals = model.get_current_visuals(need_LR=False)

        hr_img = tensor2img(visuals['HR'])
        forged_img = tensor2img(visuals['SR'])
        lr_img = tensor2img(visuals['LR'])
        cv2.imwrite('{}/{}_{}_hr.png'.format(opt["path"]["results"], current_step, idx), hr_img)
        cv2.imwrite('{}/{}_{}_sr.png'.format(opt["path"]["results"], current_step, idx), forged_img)
        cv2.imwrite('{}/{}_{}_lr.png'.format(opt["path"]["results"], current_step, idx), lr_img)
        # Calculate PSNR for the current batch
        psnr_sum += psnr(hr_img, forged_img)
        ssim_sum += ssim(hr_img, forged_img)
        num_samples += 1

    # Calculate the average PSNR
    average_psnr = psnr_sum / num_samples
    average_ssim = ssim_sum / num_samples
    # Log the validation results
    logger.info(f'Validation at step {current_step}:\tAverage PSNR: {average_psnr}')
    logger.info(f'Validation at step {current_step}:\tAverage SSIM: {average_ssim}')
    # You can add more metrics and save the validation results as needed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='image_generation.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train'], help='train', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')

    args = parser.parse_args()
    opt = parse(args)

    # Ensure the required directories exist
    Path(opt['path']['log']).mkdir(parents=True, exist_ok=True)
    Path(opt['path']['results']).mkdir(parents=True, exist_ok=True)
    Path(opt['path']['checkpoint']).mkdir(parents=True, exist_ok=True)
    # Call the training function
    train(opt)
