import torch
import argparse
from models.network_nhnet_sidd import Net
import numpy as np
import torchvision.transforms as transforms
import math
from PIL import Image
import glob
import time
import scipy.io
import os
# from torchsummary import summary

torch.set_num_threads(4)
torch.manual_seed(0)
torch.manual_seed(0)

def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # noise level for noisy image
    model_name = 'real'
    model_pool = 'model_zoo'  # fixed
    model_path = os.path.join(model_pool, model_name+'.pth')

    model = Net()
    model = torch.nn.DataParallel(model)
    # model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='BR')  # use this if BN is not merged by utils_bnorm.merge_bn(model)
    x = torch.load(model_path)
    model.load_state_dict(torch.load(model_path)['state_dict'], strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # try:
    trans = transforms.ToPILImage()
    torch.manual_seed(0)
    all_noisy_imgs = scipy.io.loadmat(args.noise_dir)['BenchmarkNoisyBlocksSrgb']
    mat_re = np.zeros_like(all_noisy_imgs)
    i_imgs,i_blocks, _,_,_ = all_noisy_imgs.shape

    for i_img in range(i_imgs):
        for i_block in range(i_blocks):
            noise = transforms.ToTensor()(Image.fromarray(all_noisy_imgs[i_img][i_block])).unsqueeze(0)
            noise = noise.to(device)
            begin = time.time()
            pred = model(noise)
            pred = pred.detach().cpu()
            mat_re[i_img][i_block] = np.array(trans(pred[0]))

    return mat_re

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='testsets/BenchmarkNoisyBlocksSrgb.mat', help='path to noise image file')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='checkpoint',
                        help='the checkpoint to eval')
    parser.add_argument('--image_size', '-sz', default=64, type=int, help='size of image')
    parser.add_argument('--model_type',default="nhnet", help='type of model : KPN, attKPN, attWKPN')
    parser.add_argument('--save_img', "-s" ,default="", type=str, help='save image in eval_img folder ')

    args = parser.parse_args()
    #
    # args.noise_dir = '/home/dell/Downloads/FullTest/noisy'
    mat_re = test(args)

    mat = scipy.io.loadmat(args.noise_dir)
    # print(mat['BenchmarkNoisyBlocksSrgb'].shape)
    del mat['BenchmarkNoisyBlocksSrgb']
    mat['DenoisedNoisyBlocksSrgb'] = mat_re
    # print(mat)
    scipy.io.savemat("SubmitSrgb.mat",mat)
