from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import activations
from keras import initializers


__all__=['BilinearLayer']

###REF https://github.com/dapurv5/keras-neural-tensor-layer/blob/master/neural_tensor_layer.py
class BilinearLayer(Layer):
    def __init__(self, units,
                 activation='linear',
                 kernel_initializer='glorot_uniform',
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.units = units  # k
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(BilinearLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert type(input_shape) is list and len(input_shape) == 2

        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 # shape=(input_shape[1][-1],input_shape[0][-1],self.units),
                                 shape=(self.units, input_shape[0][-1], input_shape[1][-1]),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        super(BilinearLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('BilinearLayer must be called on a list of tensors '
                            '(at least 2). Got: ' + str(inputs))

        f1, f2 = inputs
        M = K.dot(f1, self.W)
        M = K.expand_dims(f2, axis=1) * M
        M = self.activation(K.sum(M, axis=-1))
        return M



    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return (batch_size, self.units)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def get_config(self):
        return super(BilinearLayer, self).get_config()


if __name__ == '__main__':
    import numpy as np

    layer_bilinear = BilinearLayer(units=5, activation='linear', kernel_initializer='ones')
    f1 = K.variable(np.array([[1, 2], [2, 3], [4, 5]]))
    f2 = K.variable(np.array([[2, 3, 3, 4], [4, 5, 5, 6], [5, 6, 7, 8]]))
    layer_bilinear([f1, f2])
