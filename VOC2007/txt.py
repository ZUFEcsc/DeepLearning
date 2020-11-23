# -*- coding: utf-8 -*-
# @TIME     : 2020/11/20 21:02
# @Author   : Chen Shan
# @Email    : jacobon@foxmail.com
# @File     : txt.py
# @Software : PyCharm
import os
import random

trainval_percent = 0.66
train_percent = 0.5
xmlfilepath = 'Annotations'  # 绝对路径
txtsavepath = 'ImageSets\Main'  # 生成的四个文件的存储路径
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
ftrainval = open('ImageSets/Main/trainval.txt', 'w')
ftest = open('ImageSets/Main/test.txt', 'w')
ftrain = open('ImageSets/Main/train.txt', 'w')
fval = open('ImageSets/Main/val.txt', 'w')
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()