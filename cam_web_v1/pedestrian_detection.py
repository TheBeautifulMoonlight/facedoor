# -*- coding: utf-8 -*-

import cv2 as cv2
import random
import glob
import os
import numpy as np
from imutils.object_detection import non_max_suppression
from PIL import Image
import time


class PedestrianDetection():

    def __init__(self):
        self.hogtest = cv2.HOGDescriptor()
        self.hogtest.load('hogsvm.bin')

    def detect(self, frame):
        rects, weights = self.hogtest.detectMultiScale(frame, scale = 1.03)#参数可调

        #weight
        weights = [weight[0] for weight in weights]
        weights = np.array(weights)

        #这里返回的四个值表示的是开始始位置（x,y),长宽（xx,yy)，所以做以下处理
        for i in range(len(rects)):
            r = rects[i]
            rects[i][2] = r[0] + r[2]
            rects[i][3] = r[1] + r[3]

        choose = non_max_suppression(rects, probs = weights, overlapThresh = 0.5)#参数可调（可以把overlapThresh调小一点，也不要太小）

        for (x,y,xx,yy) in choose:
            cv2.rectangle(frame, (x, y), (xx, yy), (0, 0, 255), 2)
        
        # cv2.imshow("", frame)
        # cv2.waitKey(10)
        
        return frame

class BacksubTractor():

    def __init__(self, camera):
        self.bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
        self.es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame):
        fgmask = self.bs.apply(frame) # 背景分割器，该函数计算了前景掩码
        # 二值化阈值处理，前景掩码含有前景的白色值以及阴影的灰色值，在阈值化图像中，将非纯白色（244~255）的所有像素都设为0，而不是255
        th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        # 下面就跟基本运动检测中方法相同，识别目标，检测轮廓，在原始帧上绘制检测结果
        dilated = cv2.dilate(th, self.es, iterations=2) # 形态学膨胀
        # opencv3
        image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
        # opencv4
        # contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓        
        flag_save = 1
        for c in contours:
            if cv2.contourArea(c) > 20000: #1600:
                if flag_save:
                    print("hhh"+str(time.time()))
                    flag_save = 0
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # cv2.imshow('mog', fgmask)
        # cv2.imshow('thresh', th)
        # cv2.imshow('detection', frame)
        # cv2.waitKey(10)

        return frame

