import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
Harris角点检测:将窗口向各个方向移动(u,v)，然后计算所有差异总和。
如果在各个方向上都无明显变化，说明在平坦区域上；如果在某个方向上有变化，
说明在单一边缘上；若各个方向有变化，说明在角点上（多个边缘）
'''
def Harris角点检测():
    img = cv2.imread("opencv\\files\\cube.jpg", 1)
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blockSize = 2
    apertureSize = 3
    k = 0.04
    dst = cv2.cornerHarris(img_gray, blockSize, apertureSize, k)
    #二:领域像素大小。三：求导中使用窗口大小。四：方程中自由参数[0.04,0.06]
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j] > 120):
                cv2.circle(img, (j, i), 2, (0, 255, 0), 2)
    cv2.imshow("output", img)
    cv2.waitKey(0)
#Harris角点检测()
def Shi_Tomas角点检测():
    img = cv2.imread("opencv\\files\\cube.jpg", 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(img_gray, 35, 0.05, 10)
    print(len(corners))
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    #画出连通图
    for i, pt in enumerate(corners):
        print(pt)
        x = np.int32(pt[0][0])
        y = np.int32(pt[0][1])
        cv2.circle(img, (x, y), 3, colors[i % 6], 3)
    cv2.imshow("output", img)
    cv2.waitKey(0)
#Shi_Tomas角点检测()
def 亚像素级角点检测():
    img = cv2.imread("opencv\\files\\cube.jpg", 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(img_gray, 35, 0.05, 10)
    print(len(corners))
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    #画出连通图
    for i, pt in enumerate(corners):
        print(pt)
        x = np.int32(pt[0][0])
        y = np.int32(pt[0][1])
        cv2.circle(img, (x, y), 3, colors[i % 6], 3)
    winSize = (3, 3)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
    corners = cv2.cornerSubPix(img_gray, corners, winSize, zeroZone, criteria)
    for i in range(corners.shape[0]):
        print(" -- Refined Corner [", i, "]  (",
              corners[i, 0, 0], ",", corners[i, 0, 1], ")")
    cv2.imshow("output", img)
    cv2.waitKey(0)
亚像素级角点检测()
