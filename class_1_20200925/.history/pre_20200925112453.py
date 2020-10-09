'''
Author: your name
Date: 2020-09-25 11:14:43
LastEditTime: 2020-09-25 11:24:53
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \claas_1_20200925\pre.py
'''
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
        self.image = self.image.convert("L")
        self.image = self.image.resize((28, 28))
        self.imgdata = np.asarray(self.image)
        self.imgdata = self.imgdata.reshape([784])
        self.imgdata = self.imgdata.astype("float32") / 255
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
