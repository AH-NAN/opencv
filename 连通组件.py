import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
联通组件标记算法(connected component labeling algorithm)算法，扫描
二值图像的每个像素点，对于像素点相同而且连通的分为相同的组(group),最终得到图像的像素连通件
函数:ret,labels=cv2.connectdComponents(iamge,connectivity,ltype)
参一:二值图像，黑色背景。二：连通域，默认8连通。三：labels类型。默认CV_32S。
'''
def 连通组件():
    img = cv2.imread('opencv\\files\pill.jpg', 1)
    src = cv2.GaussianBlur(img, (3, 3), 0)
    img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('bonary', binary)
    output = cv2.connectedComponents(binary, connectivity=8, ltype=cv2.CV_32S)
    #这个output第一个元素是int，表示连通数，第二个元素就是图片（和原图大小一样），上面值表示这个像素点属于第几个组件
    num_labels = output[0]
    labels = output[1]
    #构造颜色
    colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
    #画出连通图
    h, w = img_gray.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            image[row, col] = colors[labels[row, col]]
    cv2.imshow('colored labels', image)
    cv2.waitKey(0)
#连通组件()
def 连通组件状态统计():
    img = cv2.imread('opencv\\files\\rice.jpg', 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary_ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #使用开运算取出外界噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))#通过这个函数组建卷积核
    binary = cv2.morphologyEx(binary_, cv2.MORPH_OPEN, kernel)
    cv2.imshow('binary', binary_)
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S)
    print(num_labels,stats.shape,stats[1])
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    image = np.copy(img)
    for t in range(1, num_labels, 1):
        x, y, w, h, area = stats[t]#area是面积。为什么从1开始循环，因为0对应的是原图，stats长16，num_labels=16，但实际上只有15个组件。
        #x,y是连通组件外接矩形左上角的坐标。w,h是宽高。area是面积（像素统计）
        cx, cy = centers[t]
        #标出中心位置
        cv2.circle(image, (np.int32(cx), np.int32(cy)), 2, (0, 255, 0), 2, 8, 0)
        #画出外接矩形
        cv2.rectangle(image, (x, y), (x+w, y+h), colors[t%6], 1, 8, 0)
        cv2.putText(image, "No." + str(t), (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
        print("label index %d, area of the label : %d" % (t, area))

    cv2.imshow("colored labels", image)
    print("total number : ", num_labels - 1)#因为第0元素表示原图，所以总数要-1.
    cv2.waitKey(0)
连通组件状态统计()
