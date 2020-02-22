import cv2
import numpy as np
import matplotlib.pyplot as plt
def 绘制直方图():
    img = cv2.imread('opencv\\files\dog2.jpg', 1)
    hist0 = cv2.calcHist([img], [0], None, [32], [0, 256])
    hist1 = cv2.calcHist([img], [1], None, [32], [0, 256])
    hist2 = cv2.calcHist([img], [2], None, [32], [0, 256])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist3 = cv2.calcHist([img_gray], [0], None, [32], [0, 256])
    #参二:0表示灰度,B/G/R分别用1/2/3表示.参三：表示计算的区域，None表示全部
    #参四:表示划分区段的数目，32表示一共32个区段，比如256/32=8，所以统计0-7,8-15,...共32个区段的像素
    #参五:表示像素范围
    plt.subplot(121)
    plt.plot(hist0,color="b")
    plt.plot(hist1,color="g")
    plt.plot(hist2,color="r")
    plt.plot(hist3,color="gray")
    #numpy也可以计算，且更快
    plt.subplot(122)
    hist_np = np.bincount(img_gray.ravel(), minlength=32)  #ravel将二维数组展开为一维
    plt.plot(hist_np,marker='o')
    #matplot也可以算
    plt.hist(img_gray.ravel(), 32, [0, 256])
    #如果想要计算R通道直方图
    hist_npR = np.bincount(img[:,:, 2].ravel(), minlength=32)
    plt.plot(hist_npR, marker='v')
    plt.show()
    cv2.waitKey(0)
#绘制直方图()
def 直方图均衡化():
    img = cv2.imread("opencv\\files\\sky.jpg", 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #灰度图均衡化
    equ_gray = cv2.equalizeHist(img_gray)
    #彩色图均衡化，对每个通道均衡化，再合并
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    equ_bgr = cv2.merge((bH, gH, rH))
    #输出
    gray = np.hstack((img_gray, equ_gray))
    bgr = np.hstack((img, equ_bgr))
    plt.subplot(3, 1, 1)
    plt.imshow(gray, 'gray')
    plt.subplot(3, 1, 2)
    plt.imshow(bgr)
    '''
    直方图均衡化是对整幅图像而言，会导致部分地方过亮过暗。CLAHE自适应均衡化：
    在每一个小区域内进行均衡化，使局部不会太突兀。如果有噪点的话，也会被放大。
    彩色图也要先分割，再merge。
    '''
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img_gray)
    plt.subplot(3, 1, 3)
    plt.imshow(img_gray, 'gray')
    plt.show()
    cv2.waitKey(0)
#直方图均衡化()
def 直方图2D():
    '''
    考虑每个颜色(Hue)和饱和度(Saturaion)
    同样用cv2.calsHist()计算直方图
    '''
    img = cv2.imread("opencv\\files\\j20.jpg", 1)
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    #二：[0,1]表示同时处理H,S。四:H通道180，S通道256。
    plt.imshow(hist, interpolation='nearest')
    plt.show()
#直方图2D()
def 直方图反向投影():
    roi = cv2.imread("opencv\\files\\dog_head.jpg", 1)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    #目标搜索图片
    target = cv2.imread("opencv\\files\\dog.jpg", 1)
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    #计算roi直方图
    roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    #归一化:参数为原图像和输出图像，归一化后值在0-255之间
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)
    #此处卷积可以把分散的点连在一起
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dst = cv2.filter2D(dst, -1, disc)
    print(dst.shape)
    ret, thresh = cv2.threshold(dst, 50, 255, 0)
    #使用merge变换成通道图像
    thresh = cv2.merge((thresh, thresh, thresh))
    print(thresh.shape)
    #按位操作
    print(target.shape)
    res = cv2.bitwise_and(target, thresh)
    res = np.hstack((target, thresh, res))
    #显示图像
    cv2.imshow('1', res)
    cv2.waitKey(0)
直方图反向投影()
    