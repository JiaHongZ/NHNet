import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image_sidd as util
import cv2
class DatasetSIDD(data.Dataset):
    def __init__(self, opt):
        super(DatasetSIDD, self).__init__()
        print('Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma
        self.paths_H, self.paths_L = util.get_image_paths_sidd_nhnet(opt['dataroot_H'])
        print('saddsa',self.paths_H)
    def __getitem__(self, index):
        import time
        H_path = self.paths_H[index]
        img_H = cv2.imread(H_path, cv2.IMREAD_UNCHANGED)
        # img_H = util.imread_uint(H_path, self.n_channels)

        L_path = self.paths_L[index]
        img_L = cv2.imread(L_path, cv2.IMREAD_UNCHANGED)
        # img_L = util.imread_uint(L_path, self.n_channels)
        if self.opt['phase'] == 'train':

            H, W, _ = img_H.shape

            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))

            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            mode = np.random.randint(0, 8)
            patch_H = util.augment_img(patch_H, mode=mode)
            patch_L = util.augment_img(patch_L, mode=mode)

            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(patch_L)

        else:
            img_H = util.uint2single(img_H)
            img_L = util.uint2single(img_L)

            img_L = util.single2tensor3(img_L)
            img_H = util.single2tensor3(img_H)
        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)
