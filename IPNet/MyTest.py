
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
# from model.IPNet import IPNet
from model.IPNet import IPNet
from utils.dataloader import test_dataset
import imageio


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='/home/image06/jiajia/transformer/model_transformer_aop/stokes_nopolar_v2/IPNet-29.pth')

for _data_name in ['test']:
    data_path = '/home/image06/jiajia/full-dataset/{}/'.format(_data_name)
    save_path = '/home/image06/jiajia/transformer/PVTv2_aop/'
    opt = parser.parse_args()
    model = IPNet()

    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    rgb_root = '{}/test-rgb/'.format(data_path)
    d0_root = '/home/image06/jiajia/full-dataset/test/test-aop/'
    d1_root = '/home/image06/jiajia/full-dataset/test/test-aop/'
    d2_root = '/home/image06/jiajia/full-dataset/test/test-aop/'

    gt_root = '{}/test-gt/'.format(data_path)
    test_loader = test_dataset(rgb_root, d0_root, d1_root, d2_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, dop, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        dop =dop.cuda()
        # aop = aop.cuda()

        prediction3, prediction2,prediction = model(image,dop)
        res = prediction3
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imsave(save_path+name, res)