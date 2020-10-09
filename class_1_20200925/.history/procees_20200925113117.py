'''
Author: your name
Date: 2020-09-25 11:24:06
LastEditTime: 2020-09-25 11:31:17
LastEditors: your name
Description: In User Settings Edit
FilePath: \claas_1_20200925\procees.py
'''
import numpy as np
from PIL import Image


def load_mnist():  # 读取离线的MNIST.npz文件。
    path = r'mnist.npz'  # 放置mnist.py的目录，这里默认跟本代码在同一个文件夹之下。
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


(train_image, train_label), (test_image, test_label) = load_mnist()
print(train_image.shape)
print(train_label.shape)

# 看第一张图片的数据
print(train_image[0])
# 转化为图片
im = Image.fromarray(train_image[0])
# 看第一张图片的大小
print(im.size)
# 看第一张图片
im.show()
