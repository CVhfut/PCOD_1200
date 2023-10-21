import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from model.IPNet import IPNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.func import label_edge_prediction

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def wbce_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    return wbce.mean()

def train(train_loader, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
    # size_rates = [0.75, 1, 1.25]
    size_rates = [1]
    # CE = torch.nn.BCEWithLogitsLoss()
    loss_record1, loss_record2, loss_record3 = AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images_rgb, images_dop, gts = pack
            images_rgb = Variable(images_rgb).cuda()
            images_dop = Variable(images_dop).cuda()
            # images_aop = Variable(images_aop).cuda()
            gts = Variable(gts).cuda()
            gt_edges = label_edge_prediction(gts)
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images_rgb = F.upsample(images_rgb, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                images_dop = F.upsample(images_dop, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # images_aop = F.upsample(images_aop, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gt_edges = F.upsample(gt_edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_1, lateral_map_2, lateral_map_3 = model(images_rgb, images_dop)
            # lateral_map_1 = model(images_rgb, images_dop)
            # ---- loss function ----
            loss1 = structure_loss(lateral_map_1, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss3 = wbce_loss(lateral_map_3, gt_edges)
            loss = loss1 + loss2 + loss3# TODO: try different weights for loss
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-1: {:.4f}, lateral-2: {:.4f}, lateral-3: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,loss_record1.show(),
                        loss_record2.show(),loss_record3.show()))
    save_path = 'model_transformer_aop/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), save_path + 'IPNet-%d.pth' % epoch)
        print('[Saving fold2:]', save_path + 'IPNet-%d.pth'% epoch)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=40, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=6, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=20, help='every n epochs decay learning rate')
    parser.add_argument('--rgb_path', type=str,
                        default='/home/image06/jiajia/full-dataset/train/train-rgb/', help='path to train dataset')
    # parser.add_argument('--black_path', type=str,
    #                     default='/home/jiajia-ding/Desktop/enhance_dataset/train-black/', help='path to train dataset')
    parser.add_argument('--d0_path', type=str,
                        default='/home/image06/jiajia/full-dataset/train/train-aop/', help='path to train dataset')
    parser.add_argument('--d1_path', type=str,
                        default='/home/image06/jiajia/full-dataset/train/train-aop/', help='path to train dataset')
    parser.add_argument('--d2_path', type=str,
                        default='/home/image06/jiajia/full-dataset/train/train-aop/', help='path to train dataset')
    parser.add_argument('--train_path', type=str,
                        default='/home/image06/jiajia/full-dataset/train/', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='stokes_nopolar_v2')
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = IPNet().cuda()
    # model.load_state_dict((torch.load(
    #     '/home/image06/jiajia/train-IPNet/IPNet_sevensuper_model/stokes_nopolar_v2/IPNet-29.pth')))

    # ---- flops and params ----
    # from utils.utils import CalParams/home/jiajia-ding/Desktop/rgb_polar_code/three input/MyTrain.py
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(lib, x)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    # trainfolder = {
    #     "rgb_root":'/home/jiajia-ding/Desktop/enhance_dataset/train-rgb/',
    #     "d0_root" :'/home/jiajia-ding/Desktop/enhance_dataset/train-dop/',
    #     "d1_root": '/home/jiajia-ding/Desktop/enhance_dataset/train-dop/',
    #     "d2_root": '/home/jiajia-ding/Desktop/enhance_dataset/train-dop/',
    #     "gt_root": '/home/jiajia-ding/Desktop/enhance_dataset/train-gt/',
    #     "edge": '/home/jiajia-ding/Desktop/enhance_dataset/train-edge.txt',
    # }

    rgb_root = opt.rgb_path
    # black_root = opt.black_path
    d0_root = opt.d0_path
    d1_root = opt.d1_path
    d2_root = opt.d2_path

    # a0_root = opt.a0_path
    # a1_root = opt.a1_path
    # a2_root = opt.a2_path
    gt_root = '{}/train-gt/'.format(opt.train_path)

    train_loader = get_loader(rgb_root, d0_root, d1_root, d2_root,
                              gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)

