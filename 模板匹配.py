import cv2
import numpy as np
import matplotlib.pyplot as plt
def 模板匹配():
    img = cv2.imread('opencv\\files\dog.jpg', 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    template = cv2.imread('opencv\\files\dog_head.jpg', 0)
    h, w = template.shape[:2]
    #6种匹配方法
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    #分别为:平方差匹配，归一化平方差匹配，相关匹配，归一化相关匹配，相关系数匹配，归一化相关系数匹配
    i = 1
    for meth in methods:
        img2 = img.copy()
        #匹配方法的真值
        method = eval(meth)
        res = cv2.matchTemplate(img_gray, template, method)
        print(res.shape)
        
        #res是一个匹配度的图像，大小一样，可以画出来。
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #如果是平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMAL，取最小值
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.circle(res, top_left, 10, 0, 2)
        plt.subplot(2, 6, 2 * i - 1)
        plt.imshow(res, 'gray')
        plt.title(meth+'res', fontsize=8)
        plt.xticks([]), plt.yticks([])
        #画矩形
        cv2.rectangle(img2, top_left, bottom_right, (0, 0, 255), 2)
        plt.subplot(2, 6, 2 * i)
        plt.imshow(img2, 'gray')
        plt.title(meth, fontsize=8)
        plt.xticks([]), plt.yticks([])
        i += 1
    plt.show()
#模板匹配()
def 多对象匹配():
    img = cv2.imread('opencv\\files\hearts.jpg', 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('opencv\\files\heart1.jpg', 0)
    h, w = template.shape[:2]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    #取匹配程度大于0.8的坐标
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        bottom_right = (pt[0] + w, pt[1] + h)
        cv2.rectangle(img, pt, bottom_right, (255, 0, 0), 1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
#多对象匹配()

'''
霍夫变换：用来在图像中提取直线和圆等几何形状
'''
def 标准直线霍夫变换():
    img = cv2.imread('opencv\\files\shapes.jpg')
    drawing = np.zeros(img.shape[:], dtype=np.uint8)  # 创建画板
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 50, 150)#边缘检测
    # 霍夫直线变换
    lines = cv2.HoughLines(edges, 0.8, np.pi / 180, 90)
    #参一:原图，一般边缘检测后。二：距离r的精度，越大考虑线越多。
    #三：角度a的精度，越小考虑越多线。四：累计数阈值，越小考虑越多线
    # 将检测的线画出来（极坐标）
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # 由参数空间向实际坐标点转换
        x1 = int(x0 + 1000 * (-b))#这1000是数目东西?
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0))

    cv2.imshow('hough lines', img)
    cv2.waitKey(0)
#标准直线霍夫变换()
def 概率霍夫直线变换():
    #计算量小，而且可以得到直线端点。从而绘出线段。
    img = cv2.imread('opencv\\files\shapes.jpg')
    drawing = np.zeros(img.shape[:], dtype=np.uint8)  # 创建画板
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 50, 150)

    ## 霍夫直线变换
    lines = cv2.HoughLinesP(edges, 0.8, np.pi/180, 90,
                            minLineLength=50, maxLineGap=10)
    # 画出直线
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.imshow('hough lines', img)
    cv2.waitKey(0)
#概率霍夫直线变换()
def 霍夫圆变换():
    img = cv2.imread('opencv\\files\\ig-logo.jpg', 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = img.copy()
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1,
                            10, param1=100, param2=30, minRadius=60, maxRadius=90)
    circles = np.int0(np.around(circles))

    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv2.circle(img, (i[0], i[1]), 2, (0, 255, 0), 2)
        cv2.rectangle(img, (i[0]-i[2], i[1]+i[2]),
                    (i[0]+i[2], i[1]-i[2]), (255, 0, 0), 2)
    cv2.imshow('circles', img)
    cv2.waitKey(0)
霍夫圆变换()
