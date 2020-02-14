import cv2
import numpy as np
import matplotlib.pyplot as plt
def 寻找轮廓(path):
    img = cv2.imread('path',0)
    ret, thresh = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 寻找二值化图中的轮廓
    contours, hierarchy = cv2.findContours(thresh,
                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(type(contours), '\n', len(contours))
    return img, contours
#寻找轮廓('opencv\\files\dog2.jpg')
def 绘制轮廓():
    img, contours = 寻找轮廓('opencv\\files\dog2.jpg')
    img_con=cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    #-1表示绘制所有轮廓，否则表示绘制那一条轮廓，(0,0,255)是BGR表示红色，2表示线宽
    cv2.imshow('img_con',img_con)
    k=cv2.waitKey(0)
#绘制轮廓()

#轮廓层级:最外轮廓叫0级，轮廓里的轮廓是子轮廓，高一级。
##findContours的返回值hierachy是一个包含四值的数组[Next,Previous,FirstChild,Parent]
def 轮廓层级示例():#将原图的三个内圈填色。
    img = cv2.imread('opencv\\files\circle3.jpg', 0)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #寻找轮廓,使用cv2.RETR_CCOMP方法寻找内外轮廓
    contours, hierarchy = cv2.findContours(th, cv2.RETR_CCOMP, 2)
    #找到内层轮廓，并填充
    #hierarch形状为(1,6,4),使用np.squeeze压缩一维数据，变成(6,4),(这里6是6个轮廓，4是每个轮廓4个量)
    hierarchy = np.squeeze(hierarchy)
    for i in range(len(contours)):
        #存在父轮廓,说明是里层
        if (hierarchy[i][3] != -1):
            cv2.drawContours(img, contours, i, (180, 215, 215), -1)
    cv2.imshow('result', img)
    k = cv2.waitKey(0)
#轮廓层级示例()
def 图像矩():
    img = cv2.imread('opencv\\files\\number13.jpg', 0)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, 3, 2)
    #以数字3的轮廓为例
    cnt = contours[0]
    img_color1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_color2 = np.copy(img_color1)
    cv2.drawContours(img_color1, [cnt], 0, (0, 0, 255), 2)
    cv2.imshow('img', img_color1)
    #cv2.moments得到图像矩
    M = cv2.moments(cnt)
    print(type(M), M)
    print("对象质心:cx=", int(M['m10'] / M['m00']), ",cy=", int(M['m01'] / M['m00']))
    print("轮廓面积:", M['m00'])
    #面积也可以用contourArea
    print("轮廓面积:", cv2.contourArea(cnt))
    #print("像素点个数:", cv2.countNonZero(cnt))#用来统计二值图像像素点个数
    print("轮廓周长:", cv2.arcLength(cnt, closed=True))
    cv2.waitKey(0)
#图像矩()
def 拟合外接():
    img = cv2.imread('opencv\\files\\number13.jpg', 0)
    _, thresh = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(thresh, 3, 2)
    #以数字3的轮廓为例
    cnt3 = contours[0]
    cnt1 = contours[1]#数字1轮廓
    #外接矩形
    x, y, w, h = cv2.boundingRect(cnt3)
    cv2.rectangle(img_color, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #最小外接矩形
    rect = cv2.minAreaRect(cnt3)  #rect=((中心x,中心y),(宽,高),旋转角度)
    box = np.int0(cv2.boxPoints(rect))  #矩形四个点取整,box是四个顶点的坐标
    cv2.drawContours(img_color, [box], 0, (0, 0, 255), 2)
    #最小外接圆
    (x, y), radius = cv2.minEnclosingCircle(cnt1)
    (x, y, radius) = np.int0((x, y, radius))
    cv2.circle(img_color, (x, y), radius, (0, 255, 0), 2)
    #拟合椭圆
    ellipse = cv2.fitEllipse(cnt1)
    cv2.ellipse(img_color, ellipse, (128, 128, 128), 2)
    cv2.imshow('img', img_color)
    cv2.waitKey(0)
#拟合外接()
def 形状匹配():
    img = cv2.imread('opencv\\files\shapes.jpg', 0)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, 3, 2)
    img_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cnt_a, cnt_b, cnt_c = contours[0], contours[1], contours[2]#阶梯形，小星星，大星星
    print(cv2.matchShapes(cnt_b, cnt_b, 1, 0.0))  # 0.0
    print(cv2.matchShapes(cnt_b, cnt_c, 1, 0.0))  # 3.04e-04
    print(cv2.matchShapes(cnt_b, cnt_a, 1, 0.0))  # 0.418
    img_con = cv2.drawContours(img_color, contours, -1, (255, 0, 0), 2)
    cv2.imshow('img', img_con)
    cv2.waitKey(0)
#形状匹配()
def 轮廓近似():
    #将轮廓用点更少的轮廓近似，新轮廓点数由设定精度决定。使用Douglas-Peucker算法
    #1.先找到轮廓
    img = cv2.imread('opencv\\files\\apple.jpg', 0)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, 3, 2)
    cnt = contours[3]
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_color1 = np.copy(img_color)
    img_color2 = np.copy(img_color)
    #2.多边形逼近，得到多边形角点
    approx_more = cv2.approxPolyDP(cnt, 3, True)#3表示精确度，越小越精确，True表示闭合
    approx_less = cv2.approxPolyDP(cnt, 50, True)
    #3.画出多边形
    cv2.polylines(img_color1, [approx_more], True, (255, 0, 0), 2)
    cv2.polylines(img_color2, [approx_less], True, (255, 0, 0), 2)
    cv2.imshow('img', np.hstack((img_color,img_color1,img_color2)))
    cv2.waitKey(0)
轮廓近似()
def 凸包():
    img = cv2.imread('opencv\\files\\apple.jpg', 0)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, 3, 2)
    cnt = contours[3]#3
    print(contours)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_color1 = np.copy(img_color)
    img_color2 = np.copy(img_color)
    #寻找凸包,得到凸包的角点
    hull = cv2.convexHull(cnt)  #可选参数returnPoins默认为True，返回轮廓点坐标，否则返回点索引
    print(hull)
    hull2 = cv2.convexHull(cnt, returnPoints=False)#绘制凸包要用True,绘制凸凹陷要用False
    #判断是否为凸
    print(cv2.isContourConvex(hull))
    #绘制凸包
    '''
    OpenCV提供了现成的函数来做凸面凹陷，cv2.convexityDefects().
    注意：我们要传returnPointsFalse来找凸形外壳。
    它返回了一个数组，每行包含这些值：[start point, end point, farthest point,
     approximate distance to farthest point].我们可以用图像来显示他们。
     我们画根线把start point和end point连起来。然后画一个圆在最远点。
     记住最前三个返回值是 cnt 的索引，所以我们我们得从 cnt 里拿出这些值.
    '''
    cv2.polylines(img_color1, [hull], True, (255, 0, 0), 2)
    #检测凸凹陷
    defects = cv2.convexityDefects(cnt, hull2)
    #可视化凸凹陷
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv2.line(img_color2, start, end, [0, 255, 0], 2)
        cv2.circle(img_color2, far, 3, [0, 0, 255], -1)
    cv2.imshow('img', np.hstack((img_color1,img_color2)))
    cv2.waitKey(0)
#凸包()
def 点到轮廓距离():
    img = cv2.imread('opencv\\files\\apple.jpg', 0)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, 3, 2)
    cnt = contours[3]
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #距离
    point=(100,100)
    dist = cv2.pointPolygonTest(cnt, point, True)
    print(dist)
    #可视化
    cv2.circle(img_color, point, 4, [0, 0, 255], -1)
    hull = cv2.convexHull(cnt)
    cv2.polylines(img_color, [hull], True, (255, 0, 0), 2)
    #参数3为True表示计算距离，轮廓内为正。为False只返回-1/0/1，表示相对位置
    cv2.imshow('img', img_color)
    cv2.waitKey(0)
#点到轮廓距离()


