import os
import sys 
import time 

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import random
import imageio
import cv2
from torch.utils.tensorboard import SummaryWriter
from config import get_option, map_dict, show_config  ## 

from utils.metrics import get_metrics
from utils.dataset_utils import get_train_loader, get_val_loader
from utils.model_utils import find_checkpoint, save_models
from utils.function import combined_loss
from utils.logger_utils import Logger

from model.ProFact import ProFact


def eval_step(model, epoch, val_loader,  criterion, writer):
    model.eval()
    total_loss, total_f1, total_iou = 0.0, 0.0, 0.0
    # iteration
    with tqdm(total=len(val_loader)) as pbar:
        for batch_idx, pack in enumerate(val_loader):
            batch_size = pack[0].size(0)
            images, gts = pack
            images, gts = Variable(images), Variable(gts)
            images, gts = images.cuda(), gts.cuda()
            with torch.no_grad():
                atts, dets = model(images)
                loss1 = criterion(atts, gts)
                loss2 = criterion(dets, gts)
                loss = loss1 + loss2

                total_loss += loss.item() * batch_size  
                f1, iou = get_metrics(dets, gts)
                total_f1 += f1 * batch_size 
                total_iou += iou * batch_size
                pbar.update(1)  
                pbar.set_description('Val Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(images),
                    len(val_loader.dataset),
                    100. * batch_idx / len(val_loader),
                    loss.item()))
        total_loss /= len(val_loader.dataset)  
        total_f1 /= len(val_loader.dataset) 
        total_iou /= len(val_loader.dataset) 
                            
        print('Epoch: {}, Val Loss: {:.6f}, F1: {:.6f}, mIoU: {:.6f}'.format(
                        epoch,
                        total_loss,
                        total_f1,
                        total_iou))
    writer.add_scalar('valid/loss', total_loss, epoch)
    writer.add_scalar('valid/F1', total_f1, epoch)
    writer.add_scalar('valid/mIoU', total_iou, epoch)
    return total_loss, total_f1, total_iou


def train_pipeline(cfg):

    if not os.path.exists(cfg['pth_dir']):
        os.makedirs(cfg['pth_dir'])

    # setting logger
    logger = Logger(cfg['pth_dir'] + '/' + cfg['ver'] + '.txt', is_w=True)
    writer = SummaryWriter(log_dir = os.path.join(cfg['log_dir'], cfg['ver']))

    now_time = time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime())
    logger.write("Training start time is: %s." % now_time)
    logger.write("(1). Initilization, define network and data loader")

    # define model and loss
    device = torch.device(cfg['device'])
    model = ProFact(ver = cfg['ver'], pretrained = cfg['pretrained'])  ##

    if cfg['loaded'] is True:
        load_path = cfg['load_path']
        checkpoint = torch.load(load_path, map_location = device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
        logger.write("Load checkpoint: %s." % load_path)

    model.to(device)

    criterion = combined_loss

    # define optimizer and lr scheduler
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr_init'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, \
                    T_max=cfg['total_epoch'], eta_min=cfg['lr_min'])

    data_dir = cfg['data_dir']
    train_loader = get_train_loader(data_dir, mode='train', batch_size=cfg['batch_size'], 
                                               num_workers=8, shuffle=True, img_size=cfg['img_size'])
    val_loader = get_val_loader(data_dir, mode='validation', batch_size=1,   
                                         num_workers=1, shuffle=False, img_size=cfg['img_size'])  ##

    
    train_len = len(train_loader.dataset)
    val_len = len(val_loader.dataset) 

    # print config
    show_config(
        ver=cfg['ver'], pretrained=cfg['pretrained'], loaded=cfg['loaded'], load_path=cfg['load_path'], 
        img_size=cfg['img_size'], \
        total_epoch=cfg['total_epoch'], batch_size=cfg['batch_size'], \
        init_lr=cfg['lr_init'], min_lr=cfg['lr_min'], optimizer_type=cfg['optimizer'], \
        save_dir=cfg['pth_dir'], num_train=train_len, num_val=val_len
    )
    logger.write("loss: %s." % criterion)
    logger.write("img_size: %s." % cfg['img_size'])
    logger.write("train size: %d, val size: %d." % (train_len, val_len))

    
    # training parameters
    last_epoch = 0 
    best_iou  = 0.

    # if resume, then load checkpoint file.
    if cfg['is_resume'] is True:
        logger.write(f"\tload pretrain weights in {cfg['pth_dir']}")
        checkpoint = find_checkpoint(pth_dir=cfg['pth_dir'], \
                            device=device, load_tag=cfg['resume_tag']) 
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        last_epoch = checkpoint['epoch']
        best_iou = checkpoint['best_iou']

    # training iteration 
    logger.write("(2). Training iteration:")
    for epoch in range(last_epoch + 1, cfg['total_epoch'] + 1):
        logger.write("=====================================")
        logger.write("Epoch: %3d, lr: %.7f." % (epoch, optimizer.param_groups[0]['lr']))
        
        now_train_loss, now_iter_size = 0, 0
        now_train_f1, now_train_iou = 0, 0
        train_tbar = tqdm(train_loader)
        
        model.train()
        for i, pack in enumerate(train_tbar):
            # # fetch data
            batch_size = pack[0].size(0)

            # forward and backward
            optimizer.zero_grad()
            images, gts = pack
            images = Variable(images)
            gts = Variable(gts)
            images = images.cuda()
            gts = gts.cuda()

            atts, dets = model(images)

            loss1 = criterion(atts, gts)
            loss2 = criterion(dets, gts)
            loss = loss1 + loss2
            loss.backward()

            optimizer.step()

            now_train_loss += loss * batch_size
            now_iter_size  += batch_size
            # update tdqm bar message
            train_tbar.set_postfix(loss = (now_train_loss/now_iter_size).item())

            f1, iou = get_metrics(dets, gts)
            now_train_f1 += f1 * batch_size
            now_train_iou += iou * batch_size

            if i % 30 == 0:
                iters = i + (epoch - 1) * len(train_loader)
                writer.add_scalar('train/loss', (now_train_loss/now_iter_size).item(), iters)
        
        # update learing rate scheduler
        scheduler.step()  

        logger.write('Epoch: {}, Train Loss: {:.6f}, F1: {:.5f}, mIoU: {:.5f}'.format(epoch, 
                                                                               (now_train_loss/now_iter_size).item(), 
                                                                               (now_train_f1/now_iter_size), 
                                                                               (now_train_iou/now_iter_size)))
        if epoch % 1 ==0:
            # validate 
            print("Start Validating..")
            avg_loss, avg_f1, avg_iou = eval_step(model, epoch, val_loader,  criterion, writer)

            # update best result
            if avg_iou > best_iou:
                best_iou = avg_iou
                logger.write('Epoch: {}, Save best model to best_epoch_weights.pth'.format(epoch))
                save_models(cfg['pth_dir'], epoch, best_iou, model, optimizer, scheduler, tag='best')

        save_models(cfg['pth_dir'], epoch, best_iou, model, optimizer, scheduler, tag='last')
        logger.write("\nBest iou  in epoch %d is: %3.7f, testData iou: %3.7f.\n" % (epoch, best_iou, avg_iou))       

    # log time 
    now_time = time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime())
    logger.write("End Time Is: %s." % now_time) 

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # get argment from command line.
    args = get_option()
    cfg = map_dict(args) 

    # training process 
    train_pipeline(cfg=cfg)