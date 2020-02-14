import cv2
import numpy as np
import matplotlib.pyplot as plt
def 绘制直方图():
    img = cv2.imread('opencv\\files\dog.jpg', 1)
    hist0 = cv2.calcHist(img, [0], None, [32], [0, 256])
    hist1 = cv2.calcHist(img, [1], None, [32], [0, 256])
    hist2 = cv2.calcHist(img, [2], None, [32], [0, 256])
    hist3 = cv2.calcHist(img, [40], None, [32], [0, 256])
    #参二:0表示灰度,B/G/R分别用1/2/3表示.参三：表示计算的区域，None表示全部
    #参四:表示划分区段的数目，32表示一共32个区段，比如256/32=8，所以统计0-7,8-15,...共32个区段的像素
    #参五:表示像素范围
    plt.plot(hist0)
    #plt.plot(hist1)
    #plt.plot(hist2)
    plt.plot(hist3)
    plt.show()
    cv2.waitKey(0)
绘制直方图()
