# import tkFileDialog
import os
import tkinter
import tkinter.filedialog

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import PepperNoiseFunc as pnf


def openFileWithWindow():
    root = tkinter.Tk()  # 创建一个Tkinter.Tk()实例
    root.withdraw()  # 将Tkinter.Tk()实例隐藏
    default_dir = r"文件路径"
    file_path = tkinter.filedialog.askopenfilename(title=u'选择文件', initialdir=(os.path.expanduser(default_dir)))
    return file_path


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


file_path = "lena512color.tiff"
# file_path = openFileWithWindow()
cv.namedWindow('Image_I', cv.WINDOW_KEEPRATIO)
cv.namedWindow('R1', cv.WINDOW_KEEPRATIO)
cv.namedWindow('G1', cv.WINDOW_KEEPRATIO)
cv.namedWindow('B1', cv.WINDOW_KEEPRATIO)
Im = cv.imread(file_path)
size = Im.shape
print(type(Im))
cv.resizeWindow('ImageI', size[0], size[1])
cv.imshow('Image_I', Im)
cv.imshow("R1", Im[:, :, 0])
cv.imshow("G1", Im[:, :, 1])
cv.imshow("B1", Im[:, :, 2])
temp = Im[:, :, 0]
Im[:, :, 0] = Im[:, :, 1]
Im[:, :, 1] = temp
cv.imshow("Image_I_Change", Im)
title = ['B', 'G', 'R']
m = cv.split(Im)
temp = m[0]
m[0] = m[2]
m[2] = temp
mc = []
mc.append(np.dstack((m[0], np.zeros(m[0].shape, np.uint8), np.zeros(m[0].shape, np.uint8))))
mc.append(np.dstack((np.zeros(m[1].shape, np.uint8), m[1], np.zeros(m[1].shape, np.uint8))))
mc.append(np.dstack((np.zeros(m[2].shape, np.uint8), np.zeros(m[2].shape, np.uint8), m[2])))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(mc[i])
    plt.title(title[i], fontsize=8)
    plt.xticks([])
    plt.yticks([])
plt.show()
cv.waitKey(1110)
Pepered = pnf.noise(m[2], 0.8)
print(type(Pepered))
cv.namedWindow('Pepper Noise', cv.WINDOW_KEEPRATIO)
cv.imshow('Pepper Noise', Pepered)
filtered = pnf.MedianFiler(Pepered)
filtered = filtered.astype(np.uint8)
print(filtered)
cv.namedWindow('Filtered Img', cv.WINDOW_KEEPRATIO)
cv.imshow('Filtered Img', filtered)
cv.waitKey(0)
