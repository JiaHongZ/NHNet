import os.path
import logging

import numpy as np
from datetime import datetime
from collections import OrderedDict
# from scipy.io import loadmat

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # noise level for noisy image
    model_path = '../NHNet/model_zoo/real.pth'
    from models.network_nhnet_color import Net as net
    model = net()
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    # model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='BR')  # use this if BN is not merged by utils_bnorm.merge_bn(model)
    model.load_state_dict(torch.load(model_path)['state_dict'], strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    import torchvision.transforms as transforms

    from PIL import Image
    import glob
    import time
    import scipy.io
    all_noisy_imgs = scipy.io.loadmat(r'testsets/ValidationNoisyBlocksSrgb.mat')[
        'ValidationNoisyBlocksSrgb']
    all_clean_imgs = scipy.io.loadmat(r'testsets/ValidationGtBlocksSrgb.mat')[
        'ValidationGtBlocksSrgb']

    i_imgs, i_blocks, _, _, _ = all_noisy_imgs.shape
    psnrs = []
    ssims = []
    net = model
    for i_img in range(i_imgs):
        print('i_img',i_img)
        for i_block in range(i_blocks):
            noise = transforms.ToTensor()(Image.fromarray(all_noisy_imgs[i_img][i_block])).unsqueeze(0)
            noise = noise.cuda()
            with torch.no_grad():
                pred = net(noise)
            pred = pred.detach().float().cpu()
            gt = transforms.ToTensor()((Image.fromarray(all_clean_imgs[i_img][i_block])))
            gt = gt.unsqueeze(0)
            pred = util.tensor2uint(pred)
            gt = util.tensor2uint(gt)
            psnr_t = util.calculate_psnr(pred, gt)
            ssim_t = util.calculate_ssim(pred, gt)
            psnrs.append(psnr_t)
            ssims.append(ssim_t)
    print('average', np.mean(psnrs), np.mean(ssims))

if __name__ == '__main__':
    main()
