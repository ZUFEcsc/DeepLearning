# -*- coding: utf-8 -*-
# @TIME     : 2020/9/25 9:31
# @Author   : Chen Shan
# @Email    : jacobon@foxmail.com
# @File     : MNIST_CNN.py
# @Software : PyCharm

import numpy as np
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# form PIL import Image
import cv2
#读取数据
def load_mnist(): #读取离线的MNIST.npz文件。
    path = r'mnist.npz' #放置mnist.py的目录，这里默认跟本代码在同一个文件夹之下。
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = load_mnist()


#数据预处理
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
# print(x_train.shape[0], "train samples")
# print(x_test.shape[0], "test samples")
# convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#构建模型
input_shape = (28, 28, 1)
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()

#训练
batch_size = 128
epochs = 5
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,verbose=2)

#测试
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# 加载待预测数据
def load_data():
    pre_x=[]
    path = r'F:/project/DeepLearning/img/new3.png'
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_arr = img_gray.reshape(28, 28, 1)
    img_arr = img_arr.astype('float32') / 255

    # array = np.asarray(img_gray,dtype="float32")
    # print(img_gray.shape)
    pre_x.append(img_arr)
    return pre_x

pre_x = load_data()
pre_x = np.expand_dims(pre_x, -1)
# 通过模型预测结果并输出
predict = model.predict(pre_x)
# print(predict)

predict = np.argmax(predict)
print("The result of model recognition is: "+str(predict))