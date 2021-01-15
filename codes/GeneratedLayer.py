#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/14 19:31
@File:          GeneratedLayer.py
'''

from keras.layers import Layer
from keras import initializers

class GeneratedLayer(Layer):
    def __init__(self, kernel_initializer='glorot_uniform', **kwargs):
        super(GeneratedLayer, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=input_shape[1:],
                                      initializer=self.kernel_initializer)

    def call(self, inputs, **kwargs):
        return inputs + self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(GeneratedLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))