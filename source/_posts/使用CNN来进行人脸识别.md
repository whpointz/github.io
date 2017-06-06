---
title: 使用CNN对人脸模型进行分类
date: 2017-06-03 10:17:56
tags: [图像,深度学习]
---


# 前言

研一下学期模式识别课程的期末大作业，要实现的是对CMU的人脸数据库PIE dataset做识别，在小组中我负责使用CNN的部分，在这篇博客记录一下。

数据集经过了助教的处理，下载地址： [百度云地址](http://pan.baidu.com/s/1i48ouXZ)   密码：gicp

在开始之前，经过调研决定使用TensorFlow+Keras深度学习框架。

## TensorFlow
TensorFlow是Google开发的开源深度学习框架，不知道比一年多以前（2016-3）用过的`Caffe`高到哪里去了。

╮(╯▽╰)╭

TensorFlow的优点就不说了，谁用谁知道，比较适合像我这样的入门菜鸟。
<!-- more -->
## Keras
既然有了TensorFlow，那么Keras是用来该干什么的?
我们可以将TensorFlow看做是后端，Keras是对TensorFlow接口的一个封装，使得使用起来更加的简单。除了TensorFlow，Keras支持的后端还有Theano。


## 环境配置
配置参考文档：
[Keras官方中文文档](http://keras-cn.readthedocs.io/en/latest/for_beginners/keras_windows/)

配置参考如下：

- [Anconda3 4.2.0](https://repo.continuum.io/archive/index.html) Anconda是科学计算集成Python发行版。如果电脑里面本身有Python环境，则需要在环境变量做出相应的修改。

- [Cuda 8.0](https://developer.nvidia.com/cuda-downloads) GPU编程基础工具箱

- [cudnn-8.0-win-x64-v5.1.zip](http://download.csdn.net/download/u012223520/9683325) Cuda加速库


安装命令
- TensorFlow
`$ pip install tensorflow`
- Keras
`$ pip install keras`


Samples测试

```
在一个储存Keras代码的文件夹内：
>>> git clone https://github.com/fchollet/keras.git
>>> cd keras/examples/
>>> python mnist_mlp.py
```
程序无错运行，TensorFlow+Keras安装完成。（想想Caffe的安装过程 = =）

# 速成Keras
Keras有多种网络模型，比如`Sequential模型`和`Functional模型`。

**这里只针对本次使用的CNN模型来介绍`Sequential模型`。（所以速成）**
## 快速开始Sequential模型
`Sequential模型`是多个网络层的线性堆叠。

- 构造方法：通过向`Sequential`使用`.add()`的方法一个个的将layer加入模型中。

``` python
#生成一个model
model = Sequential()
#Conv2D_1 卷积层 32个卷积核 每个卷积核大小3*3  激活函数使用relu
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
#Conv2D_2 64个卷积核 每个卷积核大小3*3 激活函数使用relu
model.add(Conv2D(64, (3, 3), activation='relu'))
#MaxPooling2D_1 池化层
model.add(MaxPooling2D(pool_size=(2, 2)))
#Dropout_1
model.add(Dropout(0.25))
#Flatten_1 将前一层输出的二维特征图flatten为一维的
model.add(Flatten())
#Dense_1 全连接有128个神经元节点，初始化方法为relu
model.add(Dense(128, activation='relu'))
#Dropout_2
model.add(Dropout(0.5))
#Dense_2 Softmax分类，输出是num_classes分类
model.add(Dense(num_classes, activation='softmax'))
```
这段代码生成的Model模型见下图：

![modelCNN](http://o9z9uibed.bkt.clouddn.com/image/20170606/201627193.png?imageslim)


## 制定输入数据的shape
`Sequential模型`的第一层需要接受一个关于输入数据的`shape`。后面的每层就可以自动的推算中间数据的shape。
传递一个`input_shape`的关键字给第一层，`input_shape`是一个tuple类型的数据。
不同的backend(TensorFlow和Theano)有着不同的input_shape格式。
在本次人脸识别使用的CNN模型中，第一层的`input_shape`可以通过图像的尺寸来算得。
``` python
from keras import backend as K
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
```


## 编译
在训练模型之前需要通过`compile`来对学习的过程进行配置。
`compile`接受三个参数：
- 损失函数`loss` 模型试图最小化的目标函数，比如`categorical_crossentropy`、`mse`
- 优化器`optimizer` eg: `rmsprop`、`adagrad`
- 指标列表`metrics` 对于分类问题，我们一般将该列表设置为`metrics=['accuracy']`，指标函数应该返回单个张量，或一个映射字典。

``` python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
```


## 训练
keras以Numpy数组作为输入数据和标签的数据类型，训练模型一般使用fit函数。
参数分别为：
- x：输入数据。如果模型只有一个输入，那么x的类型是numpy array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array。
- y：标签，numpy array。
- batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
- epochs：整数，训练的轮数，每个epoch会把训练集轮一遍。
- verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录。
- validation_data：形式为（X，Y）的tuple，指定**验证集**。此参数将覆盖validation_spilt。


``` python
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
```

#实践
## 数据对接
在`git clone`Keras之后，Keras中有一个利用CNN来训练mnist数据集的[examples](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py)。

结合上一章所写的内容我们应该可以轻易看懂这段程序。

下面如何使用我们的数据作为网络的输入呢？
``` python
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

```
写一个load_data函数将本地的数据集读进来变成`(x_train, y_train), (x_test, y_test)`对应的格式(numpy.array)即可。

下图中是mnist数据集中训练集和训练集的label，测试集和测试集的label,28为像素。
``` python
x_train.shape = (60000, 28, 28)
y_train.shape = (60000,)
x_test.shape = (10000, 28, 28)
y_test.shape = (10000,)
```
我的load_data_pie参考代码如下：
``` python
def load_data_pie(dataset_path):
    data = sio.loadmat(dataset_path)
    x_train_pie_list = []
    y_train_pie_list = []
    x_test_pie_list = []
    y_test_pie_list = []
    for i in range(len(data['gnd'])):
        if data['isTest'][i] == 0:
            x_train_pie_list.append(data['fea'][i].reshape(64,64))
            y_train_pie_list.append(data['gnd'][i])
        else:
            x_test_pie_list.append(data['fea'][i].reshape(64,64))
            y_test_pie_list.append(data['gnd'][i])

    x_train_pie = np.array(x_train_pie_list)
    y_train_pie = np.array(y_train_pie_list).reshape(len(y_train_pie_list),)
    x_test_pie = np.array(x_test_pie_list)
    y_test_pie = np.array(y_test_pie_list).reshape(len(y_test_pie_list),)

    return (x_train_pie,y_train_pie),(x_test_pie,y_test_pie)
```
## 修改参数
- `num_classes` label的数量
- `img_rows, img_cols` 图片的分辨率
- 因为不是mnist数据集，所以将二值化去掉

不出意外可以跑起来了，正确率达到了98.5%，还可以啦。

## 可视化网络模型
[参考文档](http://keras-cn.readthedocs.io/en/latest/other/visualization/)
在上图中，我贴出了一张CNN模型的图片，其使用的是可视化工具`graphviz`。

参考代码
``` python
#encoding=utf-8

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import plot_model
import mnist

batch_size = 128
num_classes = 69
epochs = 12

# input image dimensions
img_rows, img_cols = 64, 64

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data_pie('Pose29_64x64.mat')

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


def run():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    plot_model(model, to_file='model_cnn.png',show_shapes=True)#False则不显示数据维度


if __name__ == '__main__':
    run()
```


## 数据集较小的时候
[参考文档](http://keras-cn.readthedocs.io/en/latest/blog/image_classification_using_very_little_data/)

文档中的代码有些问题，这是我的参考代码
``` python
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy

datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        rescale=1./255,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=False,
        fill_mode='nearest')

img = Image.open('pose5.png')  # this is a PIL image
x = numpy.array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape( (1,)+x.shape +(1,))  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='pose5', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
```

# Last
TensorFlow和Keras对于我这样的小白真的是非常友好的深度学习工具，你完全可以在看了这篇文章之后将各种深度网络来作为一个黑盒子处理各种问题。

至少做一个大作业是没问题的啦。
