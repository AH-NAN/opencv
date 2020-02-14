import cv2
import numpy as np
import matplotlib.pyplot as plt
def 泊松融合():
    src = cv2.imread('opencv\\files\j20.jpg', 1)
    dst = cv2.imread('opencv\\files\\sky.jpg', 1)
    dst = cv2.resize(dst, (900, 700), 0, 0, cv2.INTER_AREA)
    print(src.shape, dst.shape)
    #绘制飞机的mask
    src_mask = np.zeros(src.shape, src.dtype)
    poly = np.array([[120, 110], [350, 110], [570, 150], [580, 230], [520, 320], [400, 320]], np.int32)
    #poly = np.array([[110, 120], [110, 350], [150, 570], [230, 580], [320, 520], [320, 400]], np.int32)
    cv2.fillPoly(src_mask, [poly], (255, 255, 255))
    #飞机中心在dst坐标
    center = (650, 170)
    #泊松融合
    output1 = cv2.seamlessClone(src, dst, src_mask, center, cv2.MIXED_CLONE)
    cv2.imshow('img', output1)
    cv2.waitKey(0)
泊松融合()
