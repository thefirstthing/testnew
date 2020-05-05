# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:34:08 2018

@author: zy
"""

'''
变分自编码
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from scipy.stats import norm
#https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10


# train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
# train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
# t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
# t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

mnist = input_data.read_data_sets('MNIST-data', one_hot=True)

print(type(mnist))  # <class 'tensorflow.contrib.learn.python.learn.datasets.base.Datasets'>

print('Training data shape:', mnist.train.images.shape)  # Training data shape: (55000, 784)
print('Test data shape:', mnist.test.images.shape)  # Test data shape: (10000, 784)
print('Validation data shape:', mnist.validation.images.shape)  # Validation data shape: (5000, 784)
print('Training label shape:', mnist.train.labels.shape)  # Training label shape: (55000, 10)

train_X = mnist.train.images
train_Y = mnist.train.labels
test_X = mnist.test.images
test_Y = mnist.test.labels

'''
定义网络参数
'''
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 2
learning_rate = 0.001
training_epochs = 20  # 迭代轮数
batch_size = 128  # 小批量数量大小
display_epoch = 3
show_num = 10

x = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
# 后面通过它输入分布数据，用来生成模拟样本数据
zinput = tf.placeholder(dtype=tf.float32, shape=[None, n_hidden_2])

'''
定义学习参数
'''
weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.001)),
    'mean_w1': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.001)),
    'log_sigma_w1': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.001)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], stddev=0.001)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], stddev=0.001))
}

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'mean_b1': tf.Variable(tf.zeros([n_hidden_2])),
    'log_sigma_b1': tf.Variable(tf.zeros([n_hidden_2])),
    'b2': tf.Variable(tf.zeros([n_hidden_1])),
    'b3': tf.Variable(tf.zeros([n_input]))
}

'''
定义网络结构
'''
# 第一个全连接层是由784个维度的输入样->256个维度的输出
h1 = tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
# 第二个全连接层并列了两个输出网络
z_mean = tf.add(tf.matmul(h1, weights['mean_w1']), biases['mean_b1'])
z_log_sigma_sq = tf.add(tf.matmul(h1, weights['log_sigma_w1']), biases['log_sigma_b1'])

# 然后将两个输出通过一个公式的计算，输入到以一个2节点为开始的解码部分 高斯分布样本
eps = tf.random_normal(tf.stack([tf.shape(h1)[0], n_hidden_2]), 0, 1, dtype=tf.float32)
z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))

# 解码器 由2个维度的输入->256个维度的输出
h2 = tf.nn.relu(tf.matmul(z, weights['w2']) + biases['b2'])
# 解码器 由256个维度的输入->784个维度的输出  即还原成原始输入数据
reconstruction = tf.matmul(h2, weights['w3']) + biases['b3']

# 这两个节点不属于训练中的结构，是为了生成指定数据时用的
h2out = tf.nn.relu(tf.matmul(zinput, weights['w2']) + biases['b2'])
reconstructionout = tf.matmul(h2out, weights['w3']) + biases['b3']

'''
构建模型的反向传播
'''
# 计算重建loss
# 计算原始数据和重构数据之间的损失，这里除了使用平方差代价函数，也可以使用交叉熵代价函数
reconstr_loss = 0.5 * tf.reduce_sum((reconstruction - x) ** 2)
print(reconstr_loss.shape)  # (,) 标量
# 使用KL离散度的公式
latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
print(latent_loss.shape)  # (128,)
cost = tf.reduce_mean(reconstr_loss + latent_loss)

# 定义优化器
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

num_batch = int(np.ceil(mnist.train.num_examples / batch_size))

'''
开始训练
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('开始训练')
    for epoch in range(training_epochs):
        total_cost = 0.0
        for i in range(num_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, loss = sess.run([optimizer, cost], feed_dict={x: batch_x})
            total_cost += loss

        # 打印信息
        if epoch % display_epoch == 0:
            print('Epoch {}/{}  average cost {:.9f}'.format(epoch + 1, training_epochs, total_cost / num_batch))

    print('训练完成')

    # 测试
    print('Result:', cost.eval({x: mnist.test.images}))
    # 数据可视化
    reconstruction = sess.run(reconstruction, feed_dict={x: mnist.test.images[:show_num]})
    plt.figure(figsize=(1.0 * show_num, 1 * 2))
    for i in range(show_num):
        # 原始图像
        plt.subplot(2, show_num, i + 1)
        plt.imshow(np.reshape(mnist.test.images[i], (28, 28)), cmap='gray')
        plt.axis('off')

        # 变分自编码器重构图像
        plt.subplot(2, show_num, i + show_num + 1)
        plt.imshow(np.reshape(reconstruction[i], (28, 28)), cmap='gray')
        plt.axis('off')
    plt.show()

    # 绘制均值和方差代表的二维数据
    plt.figure(figsize=(5, 4))
    # 将onehot转为一维编码
    labels = [np.argmax(y) for y in mnist.test.labels]
    mean, log_sigma = sess.run([z_mean, z_log_sigma_sq], feed_dict={x: mnist.test.images})
    plt.scatter(mean[:, 0], mean[:, 1], c=labels)
    plt.colorbar()
    plt.show()
    '''
    plt.figure(figsize=(5,4))
    plt.scatter(log_sigma[:,0],log_sigma[:,1],c=labels)
    plt.colorbar()
    plt.show()
    '''

    '''
    高斯分布取样，生成模拟数据
    '''
    n = 15  # 15 x 15的figure
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = sess.run(reconstructionout, feed_dict={zinput: z_sample})

            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.show()
