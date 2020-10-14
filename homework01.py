# -*- coding: utf-8 -*-
# @TIME     : 2020/9/25 9:31
# @Author   : Chen Shan
# @Email    : 879115657@qq.com
# @File     : homework01.py
# @Software : PyCharm

import tensorflow as tf
import cv2
import numpy as np

# 加载模型
model = tf.keras.models.load_model("model.h5")

# 加载待预测数据
def load_data():
    pre_x = []
    path = r'F:/project/DeepLearning/img/3.png'
    img = cv2.imread(path)
    # print(img.shape)
    # print(type(img))
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # print(img_gray.shape)
    img_arr = img_gray.reshape(1,784)
    img_arr = img_arr.astype('float32') / 255

    pre_x.append(img_arr)
    return pre_x

pre_x = load_data()

# 通过模型预测结果并输出
predict = model.predict(pre_x)
# print(predict)

predict = np.argmax(predict)
print("The result of model recognition is: "+str(predict))