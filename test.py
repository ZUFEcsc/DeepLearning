#coding=utf-8
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import load_model

# 数据集
(_, _), (X_test, y_test) = mnist.load_data()  # 划分MNIST训练集、测试集

# 随机数
index = random.randint(0, X_test.shape[0])
x = X_test[index]
y = y_test[index]

# 显示该数字
plt.imshow(x, cmap='gray_r')
plt.title("original {}".format(y))
plt.show()

# 加载
mymodel = load_model('mnistmodel.h5')

# 预测
x.shape = (1,784)#变成[[]]
# x = x.flatten()[None]  # 也可以用这个
predict = mymodel.predict(x)
predict = np.argmax(predict)#取最大值的位置

print('index:', index)
print('original:', y)
print('predicted:', predict)
