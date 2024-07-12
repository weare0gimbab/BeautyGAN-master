# 명령줄 인자를 받아 특정 이미지를 로드하고, 사전 훈련된 모델을
# 사용하여 메이크업 효과를 적용한 후 결과를 이미지 파일로 저장하는 독립적인 스크립트
# -*- coding: utf-8 -*-
# http://127.0.0.1:5000/
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import glob
import imageio.v2 as imageio
import cv2
import argparse

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--no_makeup', type=str, default='C:\\Users\\joeup\\Desktop\\weare0\\BeautyGAN-master\\imgs\\no_makeup\\xfsy_0405.png')
args = parser.parse_args()

def preprocess(img):
    return (img / 255.0 - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

img_size = 256
no_makeup = cv2.resize(imageio.imread(args.no_makeup), (img_size, img_size))
X_img = np.expand_dims(preprocess(no_makeup), 0)
makeups = glob.glob(os.path.join('imgs', 'makeup', '*.*'))
result = np.ones((2 * img_size, (len(makeups) + 1) * img_size, 3))
result[img_size: 2 * img_size, :img_size] = deprocess(no_makeup)

# Model path
model_path = 'C:\\Users\\joeup\\Desktop\\weare0\\BeautyGAN-master\\models'

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph(os.path.join(model_path, 'model.meta'))
saver.restore(sess, tf.train.latest_checkpoint(model_path))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

for i, makeup_path in enumerate(makeups):
    makeup = cv2.resize(imageio.imread(makeup_path), (img_size, img_size))
    Y_img = np.expand_dims(preprocess(makeup), 0)
    Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
    Xs_ = deprocess(Xs_)
    result[:img_size, (i + 1) * img_size: (i + 2) * img_size] = makeup / 255.
    result[img_size: 2 * img_size, (i + 1) * img_size: (i + 2) * img_size] = Xs_[0]

# 결과 이미지를 uint8 타입으로 변환
result = (result * 255).astype(np.uint8)

# 이미지 저장
imageio.imwrite('result.jpg', result)
