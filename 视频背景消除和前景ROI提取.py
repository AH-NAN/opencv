import cv2
import numpy as np
import matplotlib.pyplot as plt
def 背景消除():
    '''
    高斯混合模型GMM，为每个像素点选取适当的高斯分布，可以适应不同场景的照明变化
    API:cv2.createBackgroundSubtractorMOG2(int history=50,double varThreshold=16,bool detecShadows=True)
    参一：表示过往帧数，若=1，变成两帧差。二：像素与模型之间的马氏距离，
    越大，只有那些最新的像素会被归为前景，越小对光照越敏感。三：是否阴影检测，不检测的话速度会快点。
    KNN模型：cv2.createBackgroundSubtractorKNN()
    '''
    cap = cv2.VideoCapture('opencv\\files\\vtest.avi')
    fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=100, detectShadows=False)#效果更好
    #fgbg = cv2.createBackgroundSubtractorKNN(
     #   history=500, dist2Threshold=100, detectShadows=False)
    def getPerson(image, opt=1):
        #获取前景mask
        mask = fgbg.apply(image)
        #消除噪声
        line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5), (-1, -1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, line)
        cv2.imshow('mask', mask)
        #寻找轮廓
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in range(len(contours)):
            area = cv2.contourArea((contours[c]))
            if area < 150:
                continue
            rect = cv2.minAreaRect(contours[c])
            cv2.ellipse(image, rect, (255, 0, 0), 2, 8)
            cv2.circle(image, (np.int32(rect[0][0]), np.int32(rect[0][1])), 2, (0, 255, 0), 2, 8, 0)
        return image, mask
    while True:
        ret, frame = cap.read()
        #cv2.imwrite("opencv\\files\\input.png", frame)
        cv2.imshow('input', frame)
        result, m_ = getPerson(frame)
        cv2.imshow('result', result)
        k = cv2.waitKey(50) & 0xff
        if k == 27:
            cv2.imwrite("opencv\\files\\result.png",result)
            cv2.imwrite("opencv\\files\\mask.png", m_)
            break
    cap.release()
    cv2.destroyAllWindows()    
背景消除()
