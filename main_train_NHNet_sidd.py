import os
from config import Config 
opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np

import utils
from dataloaders.data_rgb import get_training_data, get_validation_data
from pdb import set_trace as stx

# from networks.nhnet_model import nhnet
from models.network_nhnet_sidd import Net

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
import glob
import time
import scipy.io
######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = r'E:\image_denoising\aaaa\SIDD_patches\train'
save_images = opt.TRAINING.SAVE_IMAGES
# 创建一个logger记录z
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Log等级总开关
# 第二步，创建一个handler，用于写入日志文件
fh = logging.FileHandler('psnr_log', mode='a')
fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
# 第三步，定义handler的输出格式
formatter = logging.Formatter(
    "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
# 第四步，将logger添加到handler里面
logger.addHandler(fh)
######### Model ###########
model_restoration = Net()
model_restoration = model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)

######### Scheduler ###########
warmup = True
if warmup:
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest    = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

######### Loss ###########
criterion = nn.MSELoss().cuda()


######### DataLoaders ###########
img_options_train = {'patch_size':opt.TRAINING.TRAIN_PS}
print(opt.OPTIM.BATCH_SIZE)
train_dataset = get_training_data(train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)

# val_dataset = get_validation_data(val_dir)
# val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

mixup = utils.MixUp_AUG()
best_psnr = 0
best_epoch = 0
best_iter = 0

eval_now = 200
print("Evaluation after every {"+str(eval_now)+"} Iterations !!!\n")

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
        
    for i, data in enumerate(tqdm(train_loader), 0):    

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0].cuda()
        input_ = data[1].cuda()

        if epoch>5:
            target, input_ = mixup.aug(target, input_)

        restored = model_restoration(input_)
        restored = torch.clamp(restored,0,1)  
        
        loss = criterion(restored, target)
    
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()

        #### Evaluation ####
        if i%eval_now==0 and i>0:
            if save_images:
                utils.mkdir(result_dir + '%d/%d'%(epoch,i))
            model_restoration.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            trans = transforms.ToPILImage()
            torch.manual_seed(0)
            all_noisy_imgs = scipy.io.loadmat(r'E:\image_denoising\zzz-finished\DRNet\DRNet\DR_new\testsets\ValidationNoisyBlocksSrgb.mat')['ValidationNoisyBlocksSrgb']
            all_clean_imgs = scipy.io.loadmat(r'E:\image_denoising\zzz-finished\DRNet\DRNet\DR_new\testsets\ValidationGtBlocksSrgb.mat')['ValidationGtBlocksSrgb']
            # noisy_path = sorted(glob.glob(args.noise_dir+ "/*.png"))
            # clean_path = [ i.replace("noisy","clean") for i in noisy_path]
            i_imgs, i_blocks, _, _, _ = all_noisy_imgs.shape
            psnrs = []
            ssims = []
            import utils.utils_image as util
            for i_img in range(i_imgs):
                print(i_img)
                for i_block in range(i_blocks):
                    noise = transforms.ToTensor()(Image.fromarray(all_noisy_imgs[i_img][i_block])).unsqueeze(0)
                    noise = noise.to(device)
                    begin = time.time()
                    with torch.no_grad():
                        pred = model_restoration(noise)
                    pred = pred.detach().float().cpu()
                    gt = transforms.ToTensor()((Image.fromarray(all_clean_imgs[i_img][i_block])))
                    gt = gt.unsqueeze(0)
                    pred = util.tensor2uint(pred)
                    gt = util.tensor2uint(gt)
                    psnr_t = util.calculate_psnr(pred, gt)
                    ssim_t = util.calculate_ssim(pred, gt)
                    psnrs.append(psnr_t)
                    ssims.append(ssim_t)
                if i_img == 3 and not i%3000 == 0: # 10000轮内记录3个图的psnr，每10000轮记录一次总的
                    logger.info('<epoch:{:3d}, iter:{:8,d}, 3 Average PSNR : {:<.4f}dB | SIMM {:<.4f}\n'.format(epoch,
                                                                                                                  i,
                                                                                                                  np.mean(
                                                                                                                      psnrs),
                                                                                                                  np.mean(
                                                                                                                     ssims)))
                    break
                else:
                    logger.info('<epoch:{:3d}, iter:{:8,d}, 3 Average PSNR : {:<.4f}dB | SIMM {:<.4f}\n'.format(epoch,
                                                                                                                i,
                                                                                                                np.mean(
                                                                                                                    psnrs),
                                                                                                                np.mean(
                                                                                                                    ssims)))
                    sign = 1
                        # logger.info('{:->4d}--> PSNR {:<.4f}db | SIMM {:<.4f}'.format(i_img, psnr_t, ssim_t))
            # testing log
            if sign == 1:
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.4f}dB | SIMM {:<.4f}\n'.format(epoch, i,  np.mean(psnrs), np.mean(ssims)))
                sign = 0
            logger.info('{:->4d}--> PSNR {:<.4f}db | SIMM {:<.4f}'.format(i_img, psnr_t, ssim_t))
            # testing log
            # print("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr))
            
            model_restoration.train()
            print('save',epoch,i)
            torch.save({'epoch': epoch,
                        'Iter': i,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_latest.pth"))

            torch.save({'epoch': epoch,
                        'Iter': i,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_epoch_{" + str(epoch)+"_"+str(i) + "}.pth"))

    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_epoch_{"+str(epoch)+"}.pth"))

