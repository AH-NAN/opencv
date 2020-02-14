import cv2
import numpy as np
import matplotlib.pyplot as plt
def 垂直边缘提取():
    #差分求图像梯度,应用特殊卷积求差分.
    img = cv2.imread("opencv\\files\sudoku.jpg", 0)
    #卷积核
    kernel = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
            #这样的核可以提取竖直边缘，转置后可提取水平边缘
    #提取竖直边缘
    dst_v = cv2.filter2D(img, -1, kernel)
    #提取水平边缘
    dst_h = cv2.filter2D(img, -1, kernel.T)
    cv2.imshow('img', np.hstack((img, dst_v, dst_h)))
    cv2.waitKey(0)
#垂直边缘提取()    

def Sobel算子():
    #sobel算子是高斯平滑和微分操作的结合体,抗噪声能力强
    #先在垂直方向计算梯度，后再水平方向计算梯度，然后求总梯度
    img = cv2.imread("opencv\\files\sudoku.jpg", 0)
    sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)  #只计算x方向
    sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)  #只计算y方向
    sobel = np.sqrt((np.square(sobelx) + np.square(sobely)))
    #Scharr算子
    scharrx = cv2.Scharr(img, -1, 1, 0)  # 只计算x方向
    scharry = cv2.Scharr(img, -1, 0, 1)  # 只计算y方向
    scharr = np.sqrt((np.square(scharrx) + np.square(scharry)))
    cv2.imshow("img", np.hstack((sobel.astype(int).astype(
        float), scharr.astype(int).astype(float))))
    cv2.waitKey(0)
    '''
    Prewitt算子k=[[-1,0,1],[-1,0,1],[-1,0,1]]
    Scharr算子(比Sobel更好用)k=[[-3,0,3],[-10,0,10],[-3,0,3]]
    '''
#Sobel算子()

#Laplacian算子是二姐边缘检测的典型代表
def Laplacian算子():
    img = cv2.imread('opencv\\files\sudoku.jpg', 0)
    laplacian = cv2.Laplacian(img, -1)
    cv2.imshow('img', np.hstack((img, laplacian)))
    cv2.waitKey(0)
#Laplacian算子()

def Canny边缘检测():
    img = cv2.imread('opencv\\files\\number13.jpg', 0)
    #阈值分割,(再检测边缘，效果更好)
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #30，70分别为低、高阈值
    edge = cv2.Canny(th, 30, 70)
    cv2.imshow('img', np.hstack((img, th, edge)))
    cv2.waitKey(0)
#Canny边缘检测()

def 图像金字塔():
    '''
    下采样：高斯内核卷积。去除偶数行列，压缩图像
    上采样：在每个方向放大两倍，新增的行列用0填充。再用内核卷积。
    '''
    img = cv2.imread('opencv\\files\dog.jpg', 1)
    #下采样,尺寸变小，分辨率降低
    lower_reso = cv2.pyrDown(img)
    #从下采样本再上采样，尺寸变大，但分辨率不会变高，因为信息丢失不会回来。
    higher_reso=cv2.pyrUp(lower_reso)
    cv2.imshow('img', img)
    cv2.imshow('lower', lower_reso)
    cv2.imshow('higher', higher_reso)
    cv2.waitKey(0)
#图像金字塔()

def 金字塔混合():
    '''
    图像金字塔的一个应用就是图像融合。
    例如，在图像缝合中，如果你需要将两幅图叠在一起，但是由于连接区域图像像素不连续性，政府图像效果看起来很差，这时就可以通过图像金字塔进行融合。
    步骤如下：
    1、读入两幅图像
    2、构建img1和img2的高斯金字塔
    3、根据高斯金字塔计算拉普拉斯金字塔(拉普拉斯金字塔适合用于重建)
    4、在拉普拉斯的每一层进行图像融合
    5、根据融合后的图像金字塔重建原始图像
    '''
    A = cv2.imread('opencv\\files\dog.jpg', 1)
    B = cv2.imread('opencv\\files\dog2.jpg', 1)
    A = cv2.resize(A, (256, 256), cv2.INTER_LINEAR)
    B = cv2.resize(B, (256, 256), cv2.INTER_LINEAR)
    print(A.shape, B.shape)
    #生成A的高斯金字塔
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpA.append(G)
        print(G.shape)
    #gpA=[A,1/4A,1/16A,...]
    #生成B的高斯金字塔
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpB.append(G)
        print(G.shape)
    lpA = [gpA[5]]  # 进行拉普拉斯金字塔处理，总共5级处理
    for i in range(5, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        # print(GE.shape)
        # print(gpA[i].shape)
        L = cv2.subtract(gpA[i-1], GE)#subtract图像相减
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]  # 进行拉普拉斯金字塔处理，总共5级处理
    for i in range(5, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i-1], GE)
        lpB.append(L)
    #左右拼接，在每一层拉普拉斯金字塔进行左右拼接
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        print('la', la.shape)
        ls = np.hstack((la[:, 0:cols // 2,:], lb[:, cols // 2:,:]))#各取一半
        LS.append(ls)
    ls_ = LS[0]
    #利用拉普拉斯金字塔，复原图像。add是图像相加
    for i in range(1, 6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    real = np.hstack((A[:,:cols // 2,:], B[:, cols // 2:,:]))
    cv2.imshow('Direct_blending---Pyramid_blending',np.hstack((real,ls_)))
    k=cv2.waitKey(0)
金字塔混合()
