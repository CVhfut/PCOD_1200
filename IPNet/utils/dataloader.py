import os
import random
import cv2
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import ImageEnhance
import torch

def cv_random_flip(img, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)


def randomRotation(image, dop, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        dop = dop.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, dop, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255

    return Image.fromarray(img)

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, rgb_root, d0_root, d1_root, d2_root,gt_root, trainsize):
        self.trainsize = trainsize
        self.images_rgb = [rgb_root + f for f in os.listdir(rgb_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.images_rgb1 = [rgb_root1 + f for f in os.listdir(rgb_root1) if f.endswith('.jpg') or f.endswith('.png')]
        self.images_dop1 = [d0_root + f for f in os.listdir(d0_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images_dop2 = [d1_root + f for f in os.listdir(d1_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images_dop3 = [d2_root + f for f in os.listdir(d2_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.images_aop1 = [a0_root + f for f in os.listdir(a0_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.images_aop2 = [a1_root + f for f in os.listdir(a1_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.images_aop3 = [a2_root + f for f in os.listdir(a2_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images_rgb = sorted(self.images_rgb)
        # self.images_rgb1 = sorted(self.images_rgb1)
        self.images_dop1 = sorted(self.images_dop1)
        self.images_dop2 = sorted(self.images_dop2)
        self.images_dop3 = sorted(self.images_dop3)
        # self.images_aop1 = sorted(self.images_aop1)
        # self.images_aop2 = sorted(self.images_aop2)
        # self.images_aop3 = sorted(self.images_aop3)
        self.gts = sorted(self.gts)
        self.filter_files_rgb()
        # self.filter_files_rgb1()
        self.filter_files_s0()
        self.filter_files_s1()
        self.filter_files_s2()
        # self.filter_files_a1()
        # self.filter_files_a2()
        # self.filter_files_a3()
        self.size = len(self.images_rgb)
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.s0_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.s1_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.s2_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image_rgb = self.rgb_loader(self.images_rgb[index])
        # image_rgb1 = self.rgb_loader(self.images_rgb1[index])
        image_dop1= self.stokes_loader(self.images_dop1[index])
        image_dop2 = self.stokes_loader(self.images_dop2[index])
        image_dop3 = self.stokes_loader(self.images_dop3[index])
        # image_aop1 = self.stokes_loader(self.images_aop1[index])
        # image_aop2 = self.stokes_loader(self.images_aop2[index])
        # image_aop3 = self.stokes_loader(self.images_aop3[index])
        gt = self.binary_loader(self.gts[index])
        DOP_cat = self.cat(image_dop1, image_dop2, image_dop3)
        image_rgb, DOP_cat, gt = randomRotation(image_rgb, DOP_cat, gt)
        # image = colorEnhance(image)
        # gt = randomPeper(gt)

        image_rgb = self.rgb_transform(image_rgb)
        # image_rgb1 = self.rgb_transform(image_rgb1)

        # AOP_cat = self.cat(image_aop1, image_aop2, image_aop3)
        image_DOP_cat = self.s0_transform(DOP_cat)
        # image_AOP_cat = self.s0_transform(AOP_cat)
        gt = self.gt_transform(gt)
        return image_rgb, image_DOP_cat, gt

    def filter_files_rgb(self):
        assert len(self.images_rgb) == len(self.gts)
        images_rgb = []
        gts = []
        for img_path, gt_path in zip(self.images_rgb, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images_rgb.append(img_path)
                gts.append(gt_path)
        self.images = images_rgb
        self.gts = gts

    def filter_files_rgb1(self):
        assert len(self.images_rgb1) == len(self.gts)
        images_rgb1= []
        gts = []
        for img_path, gt_path in zip(self.images_rgb1, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images_rgb1.append(img_path)
                gts.append(gt_path)
        self.images1 = images_rgb1
        self.gts = gts

    def filter_files_s0(self):
        assert len(self.images_dop1) == len(self.gts)
        images_polar = []
        gts = []
        for img_path, gt_path in zip(self.images_dop1, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images_polar.append(img_path)
                gts.append(gt_path)
        self.images = images_polar
        self.gts = gts

    def filter_files_s1(self):
        assert len(self.images_dop2) == len(self.gts)
        images_polar = []
        gts = []
        for img_path, gt_path in zip(self.images_dop2, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images_polar.append(img_path)
                gts.append(gt_path)
        self.images = images_polar
        self.gts = gts

    def filter_files_s2(self):
        assert len(self.images_dop3) == len(self.gts)
        images_polar = []
        gts = []
        for img_path, gt_path in zip(self.images_dop3, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images_polar.append(img_path)
                gts.append(gt_path)
        self.images = images_polar
        self.gts = gts

    def filter_files_a1(self):
        assert len(self.images_aop1) == len(self.gts)
        images_polar = []
        gts = []
        for img_path, gt_path in zip(self.images_aop1, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images_polar.append(img_path)
                gts.append(gt_path)
        self.images = images_polar
        self.gts = gts

    def filter_files_a2(self):
        assert len(self.images_aop2) == len(self.gts)
        images_polar = []
        gts = []
        for img_path, gt_path in zip(self.images_aop2, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images_polar.append(img_path)
                gts.append(gt_path)
        self.images = images_polar
        self.gts = gts

    def filter_files_a3(self):
        assert len(self.images_aop3) == len(self.gts)
        images_polar = []
        gts = []
        for img_path, gt_path in zip(self.images_aop3, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images_polar.append(img_path)
                gts.append(gt_path)
        self.images = images_polar
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def stokes_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

    def cat(self,s0, s1, s2):

        S_cat = Image.merge('RGB',(s0,s1,s2))

        return S_cat

def get_loader(rgb_root, d0_root, d1_root, d2_root, gt_root,batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):

    dataset = PolypDataset(rgb_root, d0_root, d1_root, d2_root, gt_root,trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, rgb_root, d0_root, d1_root, d2_root, gt_root, testsize):
        self.testsize = testsize
        self.images_rgb = [rgb_root + f for f in os.listdir(rgb_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.images_rgb1 = [rgb_root1 + f for f in os.listdir(rgb_root1) if f.endswith('.jpg') or f.endswith('.png')]
        self.images_dop1 = [d0_root + f for f in os.listdir(d0_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images_dop2 = [d1_root + f for f in os.listdir(d1_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images_dop3 = [d2_root + f for f in os.listdir(d2_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.images_aop1 = [a0_root + f for f in os.listdir(a0_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.images_aop2 = [a1_root + f for f in os.listdir(a1_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.images_aop3 = [a2_root + f for f in os.listdir(a2_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images_rgb = sorted(self.images_rgb)
        # self.images_rgb1 = sorted(self.images_rgb1)
        self.images_dop1 = sorted(self.images_dop1)
        self.images_dop2 = sorted(self.images_dop2)
        self.images_dop3 = sorted(self.images_dop3)
        # self.images_aop1 = sorted(self.images_aop1)
        # self.images_aop2 = sorted(self.images_aop2)
        # self.images_aop3 = sorted(self.images_aop3)
        self.gts = sorted(self.gts)
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.polar_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images_rgb)
        # self.size = len(self.images_polar)
        self.index = 0

    def load_data(self):
        image_rgb = self.rgb_loader(self.images_rgb[self.index])
        # image_rgb1 = self.rgb_loader(self.images_rgb1[self.index])
        image_dop1 = self.binary_loader(self.images_dop1[self.index])
        image_dop2 = self.binary_loader(self.images_dop2[self.index])
        image_dop3 = self.binary_loader(self.images_dop3[self.index])
        # image_aop1 = self.binary_loader(self.images_aop1[self.index])
        # image_aop2 = self.binary_loader(self.images_aop2[self.index])
        # image_aop3 = self.binary_loader(self.images_aop3[self.index])
        dop_cat = self.cat(image_dop1, image_dop2, image_dop3)
        # aop_cat = self.cat(image_aop1, image_aop2, image_aop3)
        image_rgb = self.rgb_transform(image_rgb).unsqueeze(0)
        # image_rgb1 = self.rgb_transform(image_rgb1).unsqueeze(0)
        image_dop = self.polar_transform(dop_cat).unsqueeze(0)
        # image_aop = self.polar_transform(aop_cat).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images_rgb[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image_rgb, image_dop, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def cat(self,s0, s1, s2):

        S_cat = Image.merge('RGB',(s0,s1,s2))

        return S_cat
