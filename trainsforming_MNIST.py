#GAN을 활용하여 숫자 이미지 1을 3으로, 2를 4로 변형하는 알고리즘입니다.

import cv2
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#input과 target에 대한 디렉토리를 지정합니다
train_input_dir = './image/arrange/training/input'
train_target_dir = './image/arrange/training/target'
test_input_dir = './image/arrange/test/input'
test_target_dir = './image/arrange/test/target'

#learning rate를 설정합니다.
LR = 0.0001

#initialize함수를 정의합니다.
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

#본 코드에서 활용되는 학습용 데이터는 image 데이터이므로, 이를 수치형 데이터로 바꾸는 함수를 정의합니다.
def create_train_data():
    training_input = []
    training_target = []
    for img in tqdm(os.listdir(train_input_dir)):
        input_path = os.path.join(train_input_dir, img)
        input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        training_input.append([np.array(input_img)])
    for img in tqdm(os.listdir(train_target_dir)):
        target_path = os.path.join(train_target_dir, img)
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        training_target.append([np.array(target_img)])
    # shuffle(training_data)
    training_input = np.reshape(training_input, (-1, 784))
    training_target = np.reshape(training_target, (-1, 784))
    np.save('train_data.npy', training_input)
    np.save('train_label.npy', training_target)
    return training_input, training_target

#본 코드에서 활용되는 테스트용 데이터는 image 데이터이므로, 이를 수치형 데이터로 바꾸는 함수를 정의합니다.
def create_test_data():
    test_input = []
    test_target = []
    for img in tqdm(os.listdir(test_input_dir)):
        input_path = os.path.join(test_input_dir, img)
        input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        test_input.append([np.array(input_img)])
    for img in tqdm(os.listdir(test_target_dir)):
        target_path = os.path.join(test_target_dir, img)
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        test_target.append([np.array(target_img)])
    # shuffle(test_data)
    test_input = np.reshape(test_input, (-1, 784))
    test_target = np.reshape(test_target, (-1, 784))
    np.save('test_data.npy', test_input)
    np.save('test_label.npy', test_target)
    return test_input, test_target

# Discriminator Network에 활용되는 레이어를 정의합니다.
X = tf.placeholder(tf.float32, shape=[None, 784])
D_W1 = tf.Variable(xavier_init([784,128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))
D_W2 = tf.Variable(xavier_init([128,128]))
D_b2 = tf.Variable(tf.zeros(shape=[128]))
D_W3 = tf.Variable(xavier_init([128,1]))
D_b3 = tf.Variable(tf.zeros(shape=[1]))
theta_D = [D_W1, D_W2, D_b1, D_b2, D_W3, D_b3]

# Discriminator Network 구성 함수를 정의합니다.
def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

# Generator Network에 활용되는 레이어를 정의합니다.
Z = tf.placeholder(tf.float32, shape=[None, 784])
G_W1 = tf.Variable(xavier_init([784, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))
G_W2 = tf.Variable(xavier_init([128,128]))
G_b2 = tf.Variable(tf.zeros(shape=[128]))
G_W3 = tf.Variable(xavier_init([128,784]))
G_b3 = tf.Variable(tf.zeros(shape=[784]))
output_data = tf.placeholder(tf.float32, [None, 784])
theta_G = [G_W1, G_W2, G_b1, G_b2, G_W3, G_b3]

# Discriminator Network 구성 함수를 정의합니다.
def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_log_prob = tf.matmul(G_h2, G_W3) + G_b3
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_log_prob, G_prob

#Generator에서 생성되는 결과를 시각적으로 확인하기 위한 함수를 정의합니다.
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig

#학습용 데이터와 테스트용 데이터를 생성하고, 이를 numpy array로 변환합니다.
train_data, train_label = create_train_data()
train_data = np.array(train_data).astype(np.float32)
train_label = np.array(train_label).astype(np.float32)
test_data, test_label = create_test_data()
test_data = np.array(test_data).astype(np.float32)
test_label = np.array(test_label).astype(np.float32)

#generator 함수의 outputs을  변수 G_log_prob과 G_prob에 저장합니다.
G_log_prob, G_prob = generator(Z)

#discriminaor 함수의 input을 실제 MNIST image 데이터로 설정한 다음, 해당 함수의 output을 변수 D_real과 D_logit_real에 저장합니다.
D_real, D_logit_real = discriminator(X)

#discriminaor 함수의 input을 generator를 통해 산출된 결과로 설정한 다음, 해당 함수의 output을 변수 D_fake과 D_logit_fake에 저장합니다.
D_fake, D_logit_fake = discriminator(G_log_prob)

#Generator의 cost를 아래와 같이 설정합니다.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=G_log_prob, labels=output_data))
optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(cost)

#Generator와 Discriminator의 loss를 아래와 같이 설정합니다.
D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


#GAN 모델 학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

training_epoch = 10
batch_size = 128
for epoch in range(training_epoch):
    total_batch = 1000000

    for i in range(total_batch):
        start = ((i + 1) * batch_size) - batch_size
        end = ((i + 1) * batch_size)
        batch_xs = train_data[start:end]
        batch_ys = train_label[start:end]
        G_train_feed_dict = {Z: batch_xs, output_data: batch_ys}
        if i % 1000 == 0:
            test = batch_xs[0]
            test = np.reshape(test, (-1, 784))
            test = np.array(test).astype(np.float32)
            
            samples = sess.run(G_log_prob, feed_dict={Z: test})
            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
            print(i)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: batch_ys, Z: batch_xs})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: batch_xs})
    c, _ = sess.run([cost, optimizer], feed_dict=G_train_feed_dict)
