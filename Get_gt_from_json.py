import os
import imageio
import numpy as np

from PIL import Image

json_file_path = ' '
# save_path = 'C:/Users/july/Desktop/polarimg/2020_7_1/selected/Ground_truth'
json_file = os.listdir(json_file_path)
json_file_list = [json_file_path + os.sep + file for file in json_file if file.endswith('_json')]
json_file_list = sorted(json_file_list)
index = 0
save_path = ' '

def binary_loader(path):
    threshold = 10
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.convert('L')
        # 转换为‘L’之后，只剩一个通道
        img_np = np.array(img)
        width = img.size[0]
        height = img.size[1]
        for x in range(width):
            for y in range(height):
                if img_np[y,x] != 0:
                    img_np[y,x] = 255
        img_bi = Image.fromarray(img_np.astype('uint8'))
        # print(img_bi.size)
        return img_bi

for path in json_file_list:
    gt_path = os.path.join(path + os.sep + 'label.png')
    # print(gt_path)
    # print(path)
    # name = path.split('\\')[-1].split('_json')[0]
    name = path.split('/')[-1].split('_json')[0]
    path_name = save_path + name
    print(path_name)
    # exit()
    gt = binary_loader(gt_path)
    # save_path = [save_path + str(index) + '.png']
    # imageio.imsave(str(index) + '.png', gt)
    imageio.imsave(path_name + '.png', gt)
    print(index)
    index+=1
    # print(gt.size)

print('Finish')
