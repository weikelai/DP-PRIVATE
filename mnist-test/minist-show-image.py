import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

# 加载 MNIST 数据集
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

# 定义函数展示图片
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()

# 展示训练集中的第一张图片
plot_image(x_train_image[1])

# 打印数据维度信息
print('训练集数量:', len(x_train_image))
print('测试集数量:', len(x_test_image))
print('训练集图片尺寸:', x_train_image.shape)
print('测试集图片尺寸:', x_test_image.shape)
