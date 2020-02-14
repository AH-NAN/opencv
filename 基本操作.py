import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
1. cv2.imread(): - cv2.IMREAD_COLOR：彩色图，默认值(1) - cv2.IMREAD_GRAYSCALE：灰度图(0) - 
cv2.IMREAD_UNCHANGED：包含透明通道的彩色图(-1)
2. cv2.namedWindow()创建一个窗口 - 参数1依旧是窗口的名字 - 参数2默认是cv2.WINDOW_AUTOSIZE，
表示窗口大小自适应图片，也可以设置为cv2.WINDOW_NORMAL，表示窗口大小可调 - 图片比较大的时候，可以考虑用后者
3. OpenCV中彩色图是以B-G-R通道顺序存储的，灰度图只有一个通道，图像坐标的起始点是在左上角，所以行对应的是y，列对应的是x。
   o--->x
   |
   y
'''
def open_jpg():
    img = cv2.imread('opencv\\files\dog.jpg', 0)
    #图片大小宽1024，高768
    print(type(img))
    #先定义窗口，后显示图片
    cv2.namedWindow('dog_w', cv2.WINDOW_NORMAL)

    cv2.imshow('dog_w', img)
    k = cv2.waitKey(0)
    #ord()用来获取某个字符的编码
    if k == ord('s'):
        cv2.imwrite('opencv\\files\dog_save.bmp', img)
#open_jpg()

def camera():#摄像头
    capture = cv2.VideoCapture(0)
    while True:
        #获取一帧
        ret, frame = capture.read()
        #转为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) == ord('q'):
            break
#camera()

def bitwise():#按位运算
    img1 = cv2.imread('opencv\\files\dog.jpg')
    img2 = cv2.imread('opencv\\files\ig-logo.jpg')
    #吧logo放到图像左上角
    rows, cols = img2.shape[:2]
    roi = img1[:rows,:cols]
    #创建掩码
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    #逆掩码
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    #只取出logo的像素
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    #把前景和背景合并
    dst = cv2.add(img1_bg, img2_fg)
    #roi放入原图
    img1[:rows,:cols] = dst
    #显示
    dd = np.hstack((img1_bg, dst))
    cc = np.hstack((mask, mask_inv))
    cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    cv2.namedWindow('rses', cv2.WINDOW_NORMAL)
    cv2.imshow('res', cc)
    cv2.imshow('rses', dd)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#bitwise()

#图像几何变换
##旋转，平移，缩放
##cv2.resize(),cv3.flip(),cv2.warpAffine()
def fanzhuan():#翻转变换
    img = cv2.imread('opencv\\files\dog.jpg', 1)
    dst = cv2.flip(img, -1)  #第二个参数为0表示上下翻转，>0表示左右翻转,<0表示左右上下翻转
    cv2.namedWindow('dog_w', cv2.WINDOW_NORMAL)
    cv2.imshow('dog_w', dst)
    k = cv2.waitKey(0)
#fanzhuan()
def suofang():#缩放变换
    '''
    OpenCV 提供的函数cv2.resize()可以实现,图像的尺寸可以自己手动设置虽你也可以指定缩放因子。我们可以选择择使用不同的插值方法。
    在缩放时我们推荐使用cv2.INTER_AREA虽在扩展时我们推荐使用v2.INTER_CUBIC较慢)和v2.INTER_LINEAR是cv2.INTER_LINEAR。
    '''
    img = cv2.imread('opencv\\files\dog.jpg', 1)
    dst = cv2.resize(img,(100,500),0,0,cv2.INTER_AREA)
    cv2.namedWindow('dog_w', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('dog_w', dst)
    k = cv2.waitKey(0)
#suofang()
def pingyi():#平移变换
    img = cv2.imread('opencv\\files\dog.jpg', 1)
    rows, cols = img.shape[:2]
    #定义平移矩阵。需要numpy的float32类型
    #x轴平移100，y轴平移50
    M = np.float32([[1, 0, 500], [0, 1, 100]])
    dst=cv2.warpAffine(img,M,(rows,cols))#原图像，变换矩阵，变换后的大小
    cv2.namedWindow('dog_w', cv2.WINDOW_NORMAL)
    cv2.imshow('dog_w', dst)
    k = cv2.waitKey(0)
#pingyi()
def xuanzhuan():#旋转变换
    img = cv2.imread('opencv\\files\dog.jpg', 1)
    rows, cols = img.shape[:2]
    # 45°旋转图片并缩小一半
    M = cv2.getRotationMatrix2D((col/4, rows/2 ), 45, 0.5)
    #该函数生成变换矩阵((旋转中心x，旋转中心y)，旋转角度正为逆，缩放比例)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow('rotation', dst)
    cv2.waitKey(0)
#xuanzhuan()
def toushi():  #透视变换
    img = cv2.imread('opencv\\files\dog.jpg', 1)
    rows, cols, ch = img.shape
    # 原图的四个点
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    # 输出图像的四个顶点
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    # 变换
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # 仿射
    dst = cv2.warpPerspective(img, M, (900, 500))
    # 显示
    #plt.subplot(121), plt.imshow(img), plt.title('Input')
    #plt.subplot(122), plt.imshow(dst), plt.title('Output')
    #plt.show()
    cv2.imshow('Input', img)
    cv2.imshow('Output', dst)
    cv2.waitKey(0)
#toushi()
# 图像阈值
## 使用固定阈值、自适应阈值和Otsu阈值法”二值化”图像
## OpenCV函数：cv2.threshold(), cv2.adaptiveThreshold()
def 简单阈值():
    img = cv2.imread('opencv\\files\huidutu.jpg', 0)
    # 应用5种不同的阈值方法
    #threshold(原图一般为灰度图，阈值,最大阈值,阈值方式)
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    titles = ['Original', 'BINARY', 'BINARY_INV',
              'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, th1, th2, th3, th4, th5]
    # 使用Matplotlib显示
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i], fontsize=8)
        plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
    plt.show()
#简单阈值()
def 自适应阈值():
    '''
    cv2.adaptiveThreshold()
    参数1：要处理的原图
    参数2：最大阈值，一般为255
    参数3：小区域阈值的计算方式
    ADAPTIVE_THRESH_MEAN_C：小区域内取均值
    ADAPTIVE_THRESH_GAUSSIAN_C：小区域内加权求和，权重是个高斯核
    参数4：阈值方式（跟前面讲的那5种相同）
    参数5：小区域的面积，如11就是11*11的小块
    参数6：最终阈值等于小区域计算出的阈值再减去此值
    '''
    img = cv2.imread("opencv\\files\dog.jpg", 0)
    #固定阈值
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #自适应阈值
    th2 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
    th3 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)
    titles = ['Original', 'Global(v = 127)', 'Adaptive Mean', 'Adaptive Gaussian']
    images = [img, th1, th2, th3]
    #显示
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i], fontsize=8)
        plt.xticks([]), plt.yticks([])
    plt.show()
#自适应阈值()
def Ostu阈值():
    #适合于双峰图片，自动选取阈值
    #所谓双峰图片，是指以灰度为x轴，像素点数量为y轴，形成的有两个峰的直方图
    #能自动将这两个峰分开，形成二值图片。
    img = cv2.imread("opencv\\files\\noisy.jpg", 0)
    print(img.shape)
    # 固定阈值
    ret1, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # Otsu阈值
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 先进行高斯滤波，在使用Otsu阈值
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 可视化
    images = [img, 0, th1, img, 0, th2, blur, 0, th3]
    titles = ['Original', 'Histogram', 'Global(v=100)',
            'Original', 'Histogram', "Otsu's",
            'Gaussian filtered Image', 'Histogram', "Otsu's"]
    for i in range(3):
        # 绘制原图
        plt.subplot(3, 3, i * 3 + 1)
        plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3], fontsize=8)
        plt.xticks([]), plt.yticks([])
        # 绘制直方图plt.hist，ravel函数将数组降成一维
        plt.subplot(3, 3, i * 3 + 2)
        plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1], fontsize=8)
        plt.xticks([]), plt.yticks([])
        # 绘制阈值图
        plt.subplot(3, 3, i * 3 + 3)
        plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2], fontsize=8)
        plt.xticks([]), plt.yticks([])
    plt.show()
Ostu阈值()
