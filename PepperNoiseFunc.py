import os

import numpy as np


def noise(img, snr):
    h = img.shape[0]
    w = img.shape[1]
    img1 = img.copy()
    sp = h * w  # 计算图像像素点个数
    NP = int(sp * (1 - snr))  # 计算图像椒盐噪声点个数
    for i in range(NP):
        randx = np.random.randint(1, h - 1)  # 生成一个 1 至 h-1 之间的随机整数
        randy = np.random.randint(1, w - 1)  # 生成一个 1 至 w-1 之间的随机整数
        if np.random.random() <= 0.5:  # np.random.random()生成一个 0 至 1 之间的浮点数
            img1[randx, randy] = 0
        else:
            img1[randx, randy] = 255
    return img1


def MedianFiler(img):
    [w, h] = img.shape
    out = np.zeros((w + 2, h + 2))
    out[1:w + 1, 1:h + 1] = img  # 此处自己创建的out与img类型有差别
    print(type(out))
    print(type(img))
    os.system("pause")
    for i in range(w):
        if i % 10 == 0:
            print(i)
        for j in range(h):
            if 1 < i < w and 1 < j < h:
                temp = img[i - 1:i + 2, j - 1:j + 2]
                med = np.median(temp)
                out[i, j] = med
    print('function finished')
    return out
