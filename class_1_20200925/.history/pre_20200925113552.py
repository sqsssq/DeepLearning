# -*- coding: utf-8 -*-
# @Time    : 2020/9/25 9:44
# @Author  : SanZhi
# @File    : pre.py
# @Software: PyCharm
from PIL import Image
import numpy as np
from tensorflow import keras
import tensorflow as ts


class Img:
    def __init__(self, url):
        self.url = url
        self.image = 0

    def read(self):
        im = Image.open(self.url)
        self.image = im

    def Preprocess(self):
        self.read()
        # 转灰度图
        self.image = self.image.convert("L")
        # 重定义大小
        self.image = self.image.resize((28, 28))
        # 转化为np数组
        self.imgdata = np.asarray(self.image)
        # 转化为数据要求784维
        self.imgdata = self.imgdata.reshape([784])
        # 类型转化 + 归一化
        self.imgdata = self.imgdata.astype("float32") / 255
        # 输入要求是二维数组，所以我强行加一维
        self.imgdata = np.array([self.imgdata])
        # print(self.imgdata)
        # self.image.show()

    def Predict(self):
        model = keras.models.load_model('model.h5')
        res = model.predict(self.imgdata)
        print(res)


Five = Img('5_1.png')
Five.Preprocess()
Five.Predict()