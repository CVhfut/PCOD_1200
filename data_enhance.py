from PIL import Image
from PIL import ImageEnhance
import os
import cv2
import numpy as np
from PIL import ImageFilter


def move(root_path, img_name):  # 平移，平移尺度为off
    img = Image.open(os.path.join(root_path, img_name))

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # print('1')
    # print(img)
    rows, cols, h = img.shape
    x = 50  # 水平平移
    y = 50  # 垂直平移
    M = np.float32([[1, 0, x], [0, 1, y]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    offset = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    # offset = img.offset(off,0)
    return offset


def flip_up_down(root_path, img_name):  # 翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img


def flip(root_path, img_name):  # 翻转图像 left right
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img


def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(80)  # 旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img


def randomRotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        img = image.rotate(random_angle, mode)
    return img


def randomColor(root_path, img_name):  # 随机颜色
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.open(os.path.join(root_path, img_name))
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度


def contrastEnhancement(root_path, img_name):  # 对比度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted


def brightnessEnhancement(root_path, img_name):  # 亮度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


def colorEnhancement(root_path, img_name):  # 颜色增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_col = ImageEnhance.Color(image)
    color = 1.5
    image_colored = enh_col.enhance(color)
    return image_colored


def GaussianBlur(root_path, img_name):  # gauss滤波
    image = Image.open(os.path.join(root_path, img_name))
    image_gbF = image.filter(ImageFilter.GaussianBlur(radius=2))
    return image_gbF


def randomPeper(img):  # for gt
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


imageDir = " "  # 要改变的图片的路径文件夹
saveDir = " "  # 要保存的图片的路径文件夹
os.makedirs(saveDir,exist_ok=True)
i = 1
im_path = os.listdir(imageDir)
im_path.sort()
for name in im_path:
    # print(i)
    i += 1
    image = Image.open(os.path.join(imageDir, name))
    image.save(os.path.join(saveDir, name))
    saveName = "flip_l_R_" + name  # left_right
    saveImage = flip(imageDir, name)  # 改为所用的函数名
    saveImage.save(os.path.join(saveDir, saveName))
    saveName = "flip_t_b_" + name  # up_down
    saveImage = flip_up_down(imageDir, name)  # 改为所用的函数名
    saveImage.save(os.path.join(saveDir, saveName))

    saveImage = flip(saveDir, saveName)  # 改为所用的函数名
    saveName = "flip_t_b_l_R_" + name  # both
    saveImage.save(os.path.join(saveDir, saveName))
    # savename = 'move_' + name
    # saveImg = move(imageDir, name)
    # saveImg.save(os.path.join(saveDir, savename))
