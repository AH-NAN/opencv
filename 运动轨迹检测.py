import cv2
import numpy as np
import matplotlib.pyplot as plt

def 均值漂移运动检测():
    '''
    利用直方图计算原图与区域的相似度，得到0-1的矩阵（与原图大小一样），然后把它映射到
    0-255就变成了灰度图（也叫反向投影图），值越大的地方越亮，越相似。然后将区域从当前位置向相似度增加最快的
    方向移动，多次迭代，使区域移动到下一个位置。这样就实现了跟踪。
    '''
    cap = cv2.VideoCapture("opencv\\files\\vtest.mp4")
    #读取第一帧
    ret, frame = cap.read()
    cv2.namedWindow("Demo", cv2.WINDOW_AUTOSIZE)
    #可以在图片上选择roi区域
    x, y, w, h = cv2.selectROI("Demo", frame, True, False)
    track_window = (x, y, w, h)
    #获取ROI直方图
    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    #inRange函数设置亮度阈值
    #去除低亮度像素点的影响
    #将低于和高于阈值的值设为0
    mask = cv2.inRange(hsv_roi, (26, 43, 46), (34, 255, 255))
    #计算直方图，参数为 图片（可多），通道数，蒙版区域，直方图长度，范围
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    #设置迭代的终止条件，最多迭代十次
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #直方图的反向投影
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        #均值迁移,搜索更新roi区域
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        #绘制窗口
        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.imshow("Demo", frame)
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite("opencv\\files\\"+chr(k)+".jpg ",frame)
    cv2.destroyAllWindows()
    cap.release()


均值漂移运动检测()
