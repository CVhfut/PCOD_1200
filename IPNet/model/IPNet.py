import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model.resnet import resnet18

from model.Res2Net_v1b import res2net50_v1b_26w_4s
from model.PVTv2 import *
# from model.resnet import ResNet18

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x

# -------------------最后的Unet结构----------------------
class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class ConcatOutput(nn.Module):
    def __init__(self, channel):
        super(ConcatOutput, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.output = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

    def forward(self, x1, x2, x3, x4):
        x3 = torch.cat((x3, self.conv_upsample1(self.upsample(x4))), 1)
        x3 = self.conv_cat1(x3)

        x2 = torch.cat((x2, self.conv_upsample2(self.upsample(x3))), 1)
        x2 = self.conv_cat2(x2)

        x1 = torch.cat((x1, self.conv_upsample3(self.upsample(x2))), 1)
        x1 = self.conv_cat3(x1)

        x = self.output(x1)
        return x

class Sobelxy(nn.Module):

    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):

        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class SCRN(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32):
        super(SCRN, self).__init__()
        # ---- ResNet Backbone ----
        # self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.context_encoder = pvt_v2_b4(pretrained=True)
        # ---- Receptive Field Block like module ----


        self.rfb1_1 = RFB_modified(64, channel)
        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(320, channel)
        self.rfb4_1 = RFB_modified(512, channel)

        self.output_s = ConcatOutput(channel)
        self.output_e = ConcatOutput(channel)

    def forward(self, x):

        size = x.size()[2:]  # 352 352
        # x = self.resnet.conv1(x)
        # x = self.resnet.bn1(x)
        # x = self.resnet.relu(x)
        # x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        # # ---- low-level features ----
        # x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        # x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        #
        # x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        # x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11

        endpoints = self.context_encoder.extract_endpoints(x)
        x1 = endpoints['reduction_2']  # 1 64 88 88
        x2 = endpoints['reduction_3']  # 1 128 44 44
        x3 = endpoints['reduction_4']  # 1 320 22 22
        x4 = endpoints['reduction_5']  # 1 512 11 11

        # feature abstraction
        x_s1 = self.rfb1_1(x1)  # 2 32 88 88
        x_s2 = self.rfb2_1(x2)  # 2 32 44 44
        x_s3 = self.rfb3_1(x3)  # 2 32 22 22
        x_s4 = self.rfb4_1(x4)  # 2 32 11 11

        x_e1 = self.rfb1_1(x1)
        x_e2 = self.rfb2_1(x2)
        x_e3 = self.rfb3_1(x3)
        x_e4 = self.rfb4_1(x4)

        pred_s = self.output_s(x_s1, x_s2, x_s3, x_s4)
        pred_e = self.output_e(x_e1, x_e2, x_e3, x_e4)

        pred_s = F.upsample(pred_s, size=size, mode='bilinear', align_corners=True)
        pred_e = F.upsample(pred_e, size=size, mode='bilinear', align_corners=True)

        return pred_s, pred_e, x_s1, x_s2, x_s3, x_s4


class IPNet(nn.Module):
    def __init__(self, channel=32, alpha=0.5):
        super(IPNet, self).__init__()

        self.context_encoder = pvt_v2_b4(pretrained=True)
        self.rgb_extractor = SCRN(channel=channel)
        # self.resnet_depth = ResNet18()
        self.alpha = alpha


        self.rfb1_1 = RFB_modified(64, channel)
        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(320, channel)
        self.rfb4_1 = RFB_modified(512, channel)

        self.sabel = Sobelxy(2 * channel)
        self.sabel_conv11 = nn.Conv2d(32, 32, 1, padding=0, stride=1, dilation=1, groups=1)
        self.relu = nn.RReLU()
        self.sabel_conv33 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, stride=1, dilation=1, groups=1), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 32, 1, padding=0, stride=1, dilation=1, groups=1))
        self.sabel_conv33_33 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, stride=1, dilation=1, groups=1), nn.ReLU(inplace=True),
                                         nn.Conv2d(32, 32, 3, padding=1, stride=1, dilation=1, groups=1), nn.ReLU(inplace=True),
                                         )

        self.sabel_conv1 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, stride=1, dilation=1, groups=1), nn.ReLU(inplace=True),
                                         nn.Conv2d(32, 32, 3, padding=1, stride=1, dilation=1, groups=1), nn.ReLU(inplace=True),
                                         nn.Conv2d(32, 32, 1, padding=0, stride=1, dilation=1, groups=1))

        self.upsample = nn.Sequential(nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
                                            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
                                            nn.ReLU(inplace=True),
                                            )


        self.con1 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
                                      nn.Conv2d(32, 32, 1, padding=0), nn.ReLU(inplace=True), )

        self.conv1_m = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1_m = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_m = nn.RReLU()

        self.conv2_m = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2_m = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_m = nn.RReLU()

        self.pre_m = nn.Conv2d(32, 2, 1, padding=0)
        self.vector_m = nn.AdaptiveAvgPool2d((1, 1))

        self.pred_ori = nn.Conv2d(64, 32, 1, padding=0)

        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )

        self.output = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.prediction = nn.Conv2d(32, 1, 1, padding=0)

        self.refunet = RefUnet(1, 64)

    def forward(self, x, d):
        size = x.size()[2:]  # 352 352
        pred_s, pred_e, x_s1, x_s2, x_s3, x_s4 = self.rgb_extractor(x)  # 针对的是RGB

        pred_mask_sig = torch.sigmoid(pred_s)
        dd = pred_mask_sig * d * self.alpha + d * (1 - self.alpha)

        # d_cor = self.resnet_dop.conv1(dd)
        # d_cor = self.resnet_dop.bn1(d_cor)
        # d_cor = self.resnet_dop.relu(d_cor)
        # d_cor = self.resnet_dop.maxpool(d_cor)
        #
        # d1_1 = self.resnet_dop.layer1(d_cor)  # 1 64 88 88
        # d2_1 = self.resnet_dop.layer2(d1_1)  # 128 44 44
        # d3_1 = self.resnet_dop.layer3(d2_1)  # 256 22 22
        # d4_1 = self.resnet_dop.layer4(d3_1)  # 512 11 11

        endpoints = self.context_encoder.extract_endpoints(dd)
        d1_1 = endpoints['reduction_2']   #1 64 88 88
        d2_1 = endpoints['reduction_3']   #1 128 44 44
        d3_1 = endpoints['reduction_4']   #1 320 22 22
        d4_1 = endpoints['reduction_5']   #1 512 11 11


        d1 = self.rfb1_1(d1_1)
        d2 = self.rfb2_1(d2_1)
        d3 = self.rfb3_1(d3_1)
        d4 = self.rfb4_1(d4_1)

#---------------------------------5

        img_feature5 = self.con1(x_s4)  # 32
        dop_feature5 = self.con1(d4)
        feature5_cat = torch.cat((img_feature5, dop_feature5), 1)  # 64 11 11

        feature5_cat_sabel = self.pred_ori(feature5_cat)  # 32
        feature5_cat_sabel = self.sabel_conv33_33(feature5_cat_sabel)

        feature5_cat_c = self.sabel(feature5_cat)
        feature5_cat_c = self.sabel_conv33(feature5_cat_c)
        feature5_cat_c = feature5_cat_sabel + feature5_cat_c
        feature5_cat_c = self.relu(feature5_cat_c)

        feature5 = self.relu1_m(self.bn1_m(self.conv1_m(feature5_cat_c)))
        feature5 = self.relu2_m(self.bn2_m(self.conv2_m(feature5)))
        weight5 = self.vector_m(self.pre_m(feature5)) # 1 4 11 11

        weight5 = F.softmax(weight5, dim=1)
        w52, w51 = torch.chunk(weight5, 2, 1)
        feature5 = w52 * img_feature5 + w51 * dop_feature5  #32

        feature5_cat_in = self.pred_ori(feature5_cat) #32
        prediction5 = torch.cat((feature5_cat_in, feature5), 1)
        prediction5 = self.pred_ori(prediction5)   #32

        prediction5_vis = prediction5
        prediction5_vis = F.interpolate(prediction5_vis, size=(352,352), mode='bilinear', align_corners=False)





#----------------------------------4
        img_feature5 = self.upsample(img_feature5)
        dop_feature5 = self.upsample(dop_feature5)
        x_s3 = x_s3 + img_feature5 + dop_feature5
        d3 = d3 + img_feature5 + dop_feature5
        img_feature4 = self.con1(x_s3)  # 32
        dop_feature4 = self.con1(d3)
        feature4_cat = torch.cat((img_feature4, dop_feature4), 1)  # 64

        feature4_cat_sabel = self.pred_ori(feature4_cat)  # 32
        feature4_cat_sabel = self.sabel_conv33_33(feature4_cat_sabel)

        feature4_cat_c = self.sabel(feature4_cat)
        feature4_cat_c = self.sabel_conv33(feature4_cat_c)
        feature4_cat_c = feature4_cat_sabel + feature4_cat_c
        feature4_cat_c = self.relu(feature4_cat_c)

        feature4 = self.relu1_m(self.bn1_m(self.conv1_m(feature4_cat_c)))
        feature4 = self.relu2_m(self.bn2_m(self.conv2_m(feature4)))
        weight4 = self.vector_m(self.pre_m(feature4))
        weight4 = F.softmax(weight4, dim=1)
        w42, w41 = torch.chunk(weight4, 2, 1)
        feature4 = w42 * img_feature4 + w41 * dop_feature4  # 32

        feature4_cat_in = self.pred_ori(feature4_cat)  # 32
        prediction4 = torch.cat((feature4_cat_in, feature4), 1)
        prediction4 = self.pred_ori(prediction4)  # 32


# ----------------------------------3
        img_feature4 = self.upsample(img_feature4)
        dop_feature4 = self.upsample(dop_feature4)
        x_s2 = x_s2 + img_feature4 + dop_feature4
        d2 = d2 + dop_feature4 + img_feature4
        img_feature3 = self.con1(x_s2)  # 32
        dop_feature3 = self.con1(d2)
        feature3_cat = torch.cat((img_feature3, dop_feature3), 1)  # 64

        feature3_cat_sabel = self.pred_ori(feature3_cat)  # 32
        feature3_cat_sabel = self.sabel_conv33_33(feature3_cat_sabel)

        feature3_cat_c = self.sabel(feature3_cat)
        feature3_cat_c = self.sabel_conv33(feature3_cat_c)
        feature3_cat_c = feature3_cat_sabel + feature3_cat_c
        feature3_cat_c = self.relu(feature3_cat_c)

        feature3 = self.relu1_m(self.bn1_m(self.conv1_m(feature3_cat_c)))
        feature3 = self.relu2_m(self.bn2_m(self.conv2_m(feature3)))
        weight3 = self.vector_m(self.pre_m(feature3))
        weight3 = F.softmax(weight3, dim=1)
        w32, w31 = torch.chunk(weight3, 2, 1)
        feature3 = w32 * img_feature3 + w31 * dop_feature3  # 32

        feature3_cat_in = self.pred_ori(feature3_cat)  # 32
        prediction3 = torch.cat((feature3_cat_in, feature3), 1)
        prediction3 = self.pred_ori(prediction3)  # 32


# ----------------------------------2
        img_feature3 = self.upsample(img_feature3)
        dop_feature3 = self.upsample(dop_feature3)
        x_s1 = x_s1 + img_feature3 + dop_feature3
        d1 = d1 + dop_feature3 + img_feature3
        img_feature2 = self.con1(x_s1)  # 32
        dop_feature2 = self.con1(d1)
        feature2_cat = torch.cat((img_feature2, dop_feature2), 1)  # 64

        feature2_cat_sabel = self.pred_ori(feature2_cat)  # 32
        feature2_cat_sabel = self.sabel_conv33_33(feature2_cat_sabel)

        feature2_cat_c = self.sabel(feature2_cat)
        feature2_cat_c = self.sabel_conv33(feature2_cat_c)
        feature2_cat_c = feature2_cat_sabel + feature2_cat_c
        feature2_cat_c = self.relu(feature2_cat_c)

        feature2 = self.relu1_m(self.bn1_m(self.conv1_m(feature2_cat_c)))
        feature2 = self.relu2_m(self.bn2_m(self.conv2_m(feature2)))
        weight2 = self.vector_m(self.pre_m(feature2))
        weight2 = F.softmax(weight2, dim=1)
        w22, w21 = torch.chunk(weight2, 2, 1)
        feature2 = w22 * img_feature2 + w21 * dop_feature2  # 32

        feature2_cat_in = self.pred_ori(feature2_cat)  # 32
        prediction2 = torch.cat((feature2_cat_in, feature2), 1)
        prediction2 = self.pred_ori(prediction2)  # 32

        # prediction2_vis = prediction2
        # prediction2_vis = F.interpolate(prediction5_vis, size=(352, 352), mode='bilinear', align_corners=False)


#----------------------------------------------------------------

        prediction5 = F.upsample(prediction5, size=size, mode='bilinear', align_corners=True)
        prediction4 = F.upsample(prediction4, size=size, mode='bilinear', align_corners=True)
        prediction3 = F.upsample(prediction3, size=size, mode='bilinear', align_corners=True)
        prediction2 = F.upsample(prediction2, size=size, mode='bilinear', align_corners=True)

        prediction5 = self.conv_upsample1(prediction5)
        prediction4 = torch.cat((prediction4, prediction5), 1)
        prediction4 = self.conv_cat1(prediction4)

        prediction4 = self.conv_upsample1(prediction4)
        prediction3 = torch.cat((prediction3, prediction4), 1)
        prediction3 = self.conv_cat1(prediction3)

        prediction3 = self.conv_upsample1(prediction3)
        prediction2 = torch.cat((prediction2, prediction3), 1)
        prediction2 = self.conv_cat1(prediction2)

        prediction = self.output(prediction2)
        prediction = F.upsample(prediction, size=size, mode='bilinear', align_corners=True)
        prediction = self.refunet(prediction)

        return prediction, pred_s, pred_e

if __name__ == '__main__':

    ras = IPNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    dop = torch.randn(1, 3, 352, 352).cuda()

    out = ras(input_tensor, dop)












    # writer = SummaryWriter("./logs")
    # img1 = Image.open('/home/jiajia-ding/Desktop/new_fulldataset/test/test-rgb/LUCID_TRI050S-Q_210700175__20220531103515289_image0.jpg')
    # img2 = Image.open('/home/jiajia-ding/Desktop/new_fulldataset/test/test-dop/LUCID_TRI050S-Q_210700175__20220531103515289_image0.jpg')
    # img2 = Image.merge('RGB', (img2, img2, img2))
    #
    # transforms_ = transforms.Compose([
    #     transforms.Resize((352, 352)),
    #     transforms.ToTensor()
    # ])
    # a1 = transforms_(img1)
    # a2 = transforms_(img2)
    #
    #
    # ras = IPNet()
    # ras.load_state_dict(torch.load('/home/jiajia-ding/Desktop/IPNet_Viz/IPNet/IPNet_model/stokes_nopolar_v2/IPNet-29.pth'))


    # a1 = a1.unsqueeze(0)     #输入网络是四个维度，所以要进行增加维度
    # a2 = a2.unsqueeze(0)
    # prediction, pred_s, pred_e, d1 = ras(a1, a2)
    # b = transforms.Resize(352,352)
    # d1 = b(d1)
    # print(d1.shape)  #1 32 88 88
    # print(d2.shape)  #1 32 44 44
    # print(d3.shape)  #1 32 22 22
    # print(d4.shape)  #1 32 11 11
    # for i in range(32):
        # writer.add_image('d1_%s' % i, d1.squeeze(0)[i], 0, dataformats='HW')


    # for i in range(32):
    #     writer.add_image('d2_%s' % i, d2.squeeze(0)[i], 0, dataformats='HW')
    #
    # for i in range(32):
    #     writer.add_image('d3_%s' % i, d3.squeeze(0)[i], 0, dataformats='HW')
    #
    # for i in range(32):
    #     writer.add_image('d4_%s' % i, d4.squeeze(0)[i], 0, dataformats='HW')
    # writer.add_image('d1_%s' % i, d1.squeeze(0)[31], 0, dataformats='HW')



