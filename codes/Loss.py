'''
@Author:        ZM
@Date and Time: 2020/6/13 20:48
@File:          Loss.py
'''

from keras.layers import Layer

class Loss(Layer):
    """特殊的层，用来定义复杂loss
    """
    def __init__(self, output_axis=None, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self.output_axis = output_axis

    def call(self, inputs, **kwargs):
        loss = self.compute_loss(inputs)
        self.add_loss(loss)
        if self.output_axis is None:
            return inputs
        elif isinstance(self.output_axis, list):
            return [inputs[i] for i in self.output_axis]
        else:
            return inputs[self.output_axis]

    def compute_loss(self, inputs):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        if self.output_axis is None:
            return input_shape
        elif isinstance(self.output_axis, list):
            return [input_shape[i] for i in self.output_axis]
        else:
            return input_shape[self.output_axis]

    def get_config(self):
        config = {
            'output_axis': self.output_axis,
        }
        base_config = super(Loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))