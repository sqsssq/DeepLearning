# -*- coding: utf-8 -*-
# @Time    : 2020/9/25 9:44
# @Author  : SanZhi
# @File    : pre.py
# @Software: PyCharm
from PIL import Image
import numpy as np
from tensorflow import keras
import tensorflow as ts
import os
from PIL import ImageOps


class Img:
    def __init__(self, file_dir):
        self.url = file_dir
        self.image = []
        self.imgdata = []
        self.fileName = []

    def read(self):
        """
            读入图像
        """
        for url in os.listdir(path=self.url):
            im = Image.open(self.url + "/" + url)
            self.fileName.append(url)
            self.image.append(im)

    def Preprocess(self):
        """
            预处理
        """
        self.read()
        for image in self.image:
            # 转灰度图
            image = image.convert("L")
            # 如果不是黑底白字，反转灰度图
            # image = ImageOps.invert(image)
            # 重定义大小
            image = image.resize((28, 28))
            # 转化为np数组
            imgdata = np.asarray(image)
            # 转化为数据要求784维
            imgdata = imgdata.reshape([784])
            # 类型转化 + 归一化
            imgdata = imgdata.astype("float32") / 255
            self.imgdata.append(imgdata)
        self.imgdata = np.array(self.imgdata)

    def Predict(self, imgdata):
        """
            载入模型并预测
        """
        model = keras.models.load_model('model.h5')
        predictResult = model.predict(imgdata)
        return predictResult

    def Solove(self):
        print(self.imgdata)
        Result = self.Predict(self.imgdata)
        out = []
        for index, l in enumerate(Result):
            max_percent = -1
            max_index = -1
            for i, v in enumerate(l):
                if v > max_percent:
                    max_percent = v
                    max_index = i
            out.append({
                "max_index": max_index,
                "max_percent": max_percent
            })
        # print(out)
        for i, v in enumerate(out):
            print(self.fileName[i] + " is " + str(v['max_index']))


Five = Img('./img')
Five.Preprocess()
Five.Solove()
