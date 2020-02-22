import cv2
import numpy as np
import matplotlib.pyplot as plt
def 删除静止点的光流检测():
    '''
    cv2.calcOpticalPyrLK():输入参数
    prevImg:上一帧图片；nextImg：当前图片
    presPts：上一帧特征点向量；nextPts：与返回值中的nextPts相同
    status：与返回值中的status相同；err：与返回值中的err相同
    winSize：计算局部连续运动的窗口尺寸（图像金字塔中），默认（21，21）
    maxLevel：图像金字塔层数，0表示不同金字塔，默认3
    criteria：寻找光流迭代终止的条件
    flags：有两个宏，表示两种计算方法。OPTFLOW_USE_INITIAL_FLOW：使用估计值作为寻找到的初始光流
            OPTFLOW_LK_GET_MIN_EIGENVALS：使用最小特征值作为误差测量，default=0
    minEigThreshold：该算法计算光流方程2x2规范化矩阵最小特征值，除以窗口像素数，如果小于minEigThreshold，
        则过滤掉相应功能，不会处理该光流，因为它允许删除坏点并获得性能提升，default=1e-4
    返回参数：
    nextPts：输出一个二维点的向量，该向量用来作为光流算法的输入特征点，也是光流算法在当前帧找到特征点的新位置
    status：标志，在当前帧发现特征点标志1，否则0
    err：向量中每个特征对于的错误率
    '''
    cap = cv2.VideoCapture("opencv\\files\\vtest.avi")
    #角点检测参数
    feature_params = dict(maxCorners=100,
        qualityLevel=0.1, minDistance=7, blockSize=7)
    #KLT光流参数
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.02))
    #随机颜色
    color = np.random.randint(0, 255, (100, 3))
    #读取第一帧
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None,**feature_params,useHarrisDetector=False,k=0.04)#p0是角点也是特征点
    good_ini = p0.copy()#ini初始点
    def caldist(a, b, c, d):
        return abs(a - b) + abs(b - d)
    mask = np.zeros_like(old_frame)
    #光流跟踪
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #计算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params)
        #根据状态选择
        good_new = p1[st == 1]#st=1表示找到了特征点
        good_old = p0[st == 1]
        #删除静止点
        k = 0
        for i, (new0, old0) in enumerate(zip(good_new, good_old)):
            a0, b0 = new0.ravel()
            c0, d0 = old0.ravel()
            dist = caldist(a0, b0, c0, d0)
            if dist > 2:
                good_new[k] = good_new[i]
                good_old[k] = good_old[i]
                good_ini[k] = good_ini[i]
                k += 1
        #提取动态点
        good_ini = good_ini[:k]
        good_new = good_new[:k]
        good_old = good_old[:k]
        #绘制跟踪线
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        cv2.imshow('frame', cv2.add(frame, mask))
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cv2.imwrite('opencv\\files\\flow.jpg', cv2.add(frame, mask))
            break
        #更新
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        if good_ini.shape[0] < 40:
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            good_ini = p0.copy()
#删除静止点的光流检测()
def 反向检测的光流分析():
    cap = cv2.VideoCapture('opencv\\files\\vtest.avi')
    #角点检测参数  
    feature_params=dict(maxCorners=100,qualityLevel=0.1)

                
