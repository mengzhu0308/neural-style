#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/14 20:19
@File:          train.py
'''

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import Model, initializers
from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.models import load_model

from Loss import Loss
from GeneratedLayer import GeneratedLayer

def content_loss(content_out, generated_content_out):
    return K.mean(K.square(content_out - generated_content_out))

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = K.shape(input_tensor)
    num_locations = K.cast(input_shape[1] * input_shape[2], 'float32')

    return result / num_locations

def style_loss(style_outs, generated_style_outs):
    loss = sum([K.mean(K.square(gram_matrix(style_out) - gram_matrix(generated_style_out)))
                for (style_out, generated_style_out) in zip(style_outs, generated_style_outs)])
    loss /= len(style_outs)
    return loss

content_weight = 1e4
style_weight = 1e-3
total_variation_weight = 30

class NeuralStyleLoss(Loss):
    def compute_loss(self, inputs):
        content_out, generated_content_out, generated_x = inputs[0], inputs[1], inputs[-1]
        style_outs, generated_style_outs = inputs[2:7], inputs[7:12]

        loss = content_weight * content_loss(content_out, generated_content_out)
        loss += style_weight * style_loss(style_outs, generated_style_outs)
        loss += total_variation_weight * tf.image.total_variation(generated_x)

        return loss

input_shape = (224, 224, 3)
mean = [103.939, 116.779, 123.68]

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, input_shape[:2])
    img = img.astype('float32')
    for i in range(3):
        img[..., i] -= mean[i]
    return img

def deprocess_image(img):
    for i in range(3):
        img[..., i] += mean[i]
    img = np.clip(np.round(img), 0, 255).astype('uint8')
    cv2.imwrite('images/generated_image.jpg', img)

content_image_path = 'images/content_image.jpg'
style_image_path = 'images/style_image.jpg'
content_image = preprocess_image(content_image_path)
style_image = preprocess_image(style_image_path)
generated_image = np.zeros(input_shape, dtype='float32')

VGG19 = load_model('VGG19_ImageNet.model', compile=False)
VGG19.trainable = False

content_layer = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_model = Model(VGG19.input, [VGG19.get_layer(name).output for name in content_layer])
style_model = Model(VGG19.input, [VGG19.get_layer(name).output for name in style_layers])
content_model.trainable = False
style_model.trainable = False

content_input = Input(shape=input_shape, dtype='float32')
style_input = Input(shape=input_shape, dtype='float32')
generated_input = Input(shape=input_shape, dtype='float32')
generated_x = GeneratedLayer(kernel_initializer=initializers.constant(value=content_image))(generated_input)

content_out = content_model(content_input)
style_outs = style_model(style_input)
generated_content_out = content_model(generated_x)
generated_style_outs = style_model(generated_x)

out = NeuralStyleLoss(output_axis=-1)([content_out, generated_content_out, *style_outs, *generated_style_outs, generated_x])
train_model = Model([content_input, style_input, generated_input], out)
train_model.compile(Adam(learning_rate=10))
train_images = [np.expand_dims(content_image, 0), np.expand_dims(style_image, 0), np.expand_dims(generated_image, 0)]

def evaluate(model):
    out = model.predict_on_batch(train_images)[0]
    deprocess_image(out)

class Evaluator(Callback):
    def __init__(self):
        super(Evaluator, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        evaluate(self.model)

evaluator = Evaluator()

train_model.fit(
    x=train_images, batch_size=1, epochs=500, callbacks=[evaluator]
)