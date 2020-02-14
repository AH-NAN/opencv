import cv2
import numpy as np
import matplotlib.pyplot as plt
def 卷积2D():
    img = cv2.imread('opencv\\files\dog.jpg', 1)
    #定义卷积核
    kernel = np.ones((3, 3), np.float32) / 10
    #卷积操作
    dst = cv2.filter2D(img, -1, kernel)#-1表示通道数不变
    cv2.namedWindow('dog_w', cv2.WINDOW_NORMAL)
    cv2.imshow("dog_w", img)
    k = cv2.waitKey(0)
#卷积2D()
def 简单滤波():
    img = cv2.imread('opencv\\files\dog.jpg', 1)
    #均值滤波，如3x3，可以看出kernel=1/9[[1,1,1],[1,1,1],[1,1,1]
    mean = cv2.blur(img, (3, 3))
    #方框滤波,几乎一样，差个系数,kernel=a[[1,1,1],[1,1,1],[1,1,1]]
    box = cv2.boxFilter(img, -1, (3, 3), normalize=True)#noemalize=True时，就是均值滤波，否则a=1
    #高斯滤波
    gaussian = cv2.GaussianBlur(img, (3, 3), 1)  #第三个参数越大，模糊效果越明显
    #中值滤波
    median = cv2.medianBlur(img, 3)
    #双边滤波
    bilateral = cv2.bilateralFilter(img, 5, 75, 75)#两个75是高斯核函数的标准差
    titles = ['Original', 'Mean', 'Box',
                'Gaussian', 'Median', 'Bilateral']
    images = [img, mean, box, gaussian, median, bilateral]
    #显示
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i], fontsize=8)
        plt.xticks([]), plt.yticks([])
    plt.show()
#简单滤波()

#形态学操作
##包括膨胀，腐蚀，开运算，闭运算
def 形态学操作():
    img = cv2.imread("opencv\\files\ig-logo.jpg", 0)
    #腐蚀.取局部最小值，会使图片变细,黑的变细。
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel)  # 腐蚀
    '''
    这个kernel也叫结构元素，可用函数cv2.getStructuringElement()生成
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # 矩形结构
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # 椭圆结构
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)) # 十字结构
    '''
    #膨胀，去局部最大值，会使图片变胖,黑的变胖。
    dilation = cv2.dilate(img, kernel)
    #开运算，先腐蚀后膨胀。可以分离物体，消除小白块。
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #闭运算，先膨胀后腐蚀。消除小黑快。
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #形态学梯度,dilation-erosion,膨胀图减去腐蚀图，可得到轮廓
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    #顶帽，原图减去开运算:src-opening。
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    #黑帽,闭运算减去原图：closing-src。
    blackhat=cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    titles = ['Original', 'erosion', 'dilation',
              'opening', 'closing', 'gradient', 'tophat', 'blackhat']
    images = [img, erosion, dilation, opening,
             closing, gradient, tophat, blackhat]
    #显示
    for i in range(8):
        plt.subplot(2, 4, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i], fontsize=8)
        plt.xticks([]), plt.yticks([])
    plt.show()
形态学操作()
