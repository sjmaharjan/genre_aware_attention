from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import activations
from keras import initializers

__all__=['AttentionMLP','MLPAttention']

class AttentionMLP(Layer):
    """
    Genre Aware Attention Model

    """
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='ones',
                 v_initializer='glorot_uniform',
                 Wg_initializer='glorot_uniform',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.v_initializer = initializers.get(v_initializer)
        self.Wg_initializer = initializers.get(Wg_initializer)
        self.supports_masking = True
        super(AttentionMLP, self).__init__(**kwargs)

    def build(self, input_shape):
        assert type(input_shape) is list and len(input_shape) == 2
        # W: (EMBED_SIZE, units)
        # Wg:(GENRE_EMB_SIZE, units)
        # b: (units,)
        # v: (units,)

        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[0][-1], self.units),
                                 initializer=self.kernel_initializer,
                                 trainable=True)

        self.Wg = self.add_weight(name="W_g{:s}".format(self.name),
                                  shape=(input_shape[1][-1], self.units),
                                  initializer=self.Wg_initializer,
                                  trainable=True)

        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(self.units,),
                                 initializer=self.bias_initializer,
                                 trainable=True)

        self.v = self.add_weight(name="v_{:s}".format(self.name),
                                 shape=(self.units,),
                                 initializer=self.v_initializer,
                                 trainable=True)

        super(AttentionMLP, self).build(input_shape)

    def call(self, xs, mask=None):
        # input: [x, u]
        # x: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # g: (BATCH_SIZE, 1,GENRE_EMB_SIZE)

        x, g = xs
        g=K.squeeze(g,axis=1)
        atten_g = K.expand_dims(K.dot(g, self.Wg), axis=1)
        et = self.activation(K.dot(x, self.W) + atten_g + self.b)
        # print("Before dot et:", et.shape.eval())
        et =  K.dot(et, self.v)
        at = K.softmax(et)
        if mask is not None and mask[0] is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        # print(ot.eval())
        return K.sum(ot, axis=1)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def compute_output_shape(self, input_shape):
        # output shape: (BATCH_SIZE, EMBED_SIZE)
        return (input_shape[0][0], input_shape[0][-1])

    def get_config(self):
        return super(AttentionMLP, self).get_config()




class MLPAttention(Layer):
    """
    MLP attention 

    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='ones',
                 v_initializer='glorot_uniform',
                 #Wg_initializer='glorot_uniform',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.v_initializer = initializers.get(v_initializer)
       # self.Wg_initializer = initializers.get(Wg_initializer)
        self.supports_masking = True
        super(MLPAttention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.units),
                                 initializer=self.kernel_initializer,
                                 trainable=True)



        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(self.units,),
                                 initializer=self.bias_initializer,
                                 trainable=True)

        self.v = self.add_weight(name="v_{:s}".format(self.name),
                                 shape=(self.units,),
                                 initializer=self.v_initializer,
                                 trainable=True)

        super(MLPAttention, self).build(input_shape)

    def call(self, xs, mask=None):
        x= xs
        et = self.activation(K.dot(x, self.W) + self.b)
        # print("Before dot et:", et.shape.eval())
        et =  K.dot(et, self.v)
        at = K.softmax(et)
        if mask is not None and mask[0] is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        # print(ot.eval())
        return K.sum(ot, axis=1)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def compute_output_shape(self, input_shape):
        # output shape: (BATCH_SIZE, EMBED_SIZE)
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(MLPAttention, self).get_config()




if __name__ == '__main__':
    import numpy as np

    layer_attention = AttentionMLP(units=6, activation='linear', kernel_initializer='ones', bias_initializer='ones')
    lstm_output = K.variable(np.array([[[1, 2, 1, 2], [2, 3, 3, 5], [4, 5, 5, 7]],
                                       [[2, 3, 6, 9], [4, 5, 2, 7], [5, 6, 7, 9]]]))

    genre = K.variable(np.array([[[1, 0, 1, 2, 3]], [[1, 0, 5, 7, 7]]]))

    layer_attention([lstm_output, genre])

    '''

    Using Theano backend.
    [[[ 1.  2.  1.  2.]
      [ 2.  3.  3.  5.]
      [ 4.  5.  5.  7.]]

     [[ 2.  3.  6.  9.]
      [ 4.  5.  2.  7.]
      [ 5.  6.  7.  9.]]]
    [[ 1.  0.  1.  2.  3.]
     [ 1.  0.  5.  7.  7.]]
    shape MLP: [2 3 6]
    shape_genre [2 1 6]
    [[[  6.   6.   6.   6.   6.   6.]
      [ 13.  13.  13.  13.  13.  13.]
      [ 21.  21.  21.  21.  21.  21.]]

     [[ 20.  20.  20.  20.  20.  20.]
      [ 18.  18.  18.  18.  18.  18.]
      [ 27.  27.  27.  27.  27.  27.]]]
    [[[  7.   7.   7.   7.   7.   7.]]

     [[ 20.  20.  20.  20.  20.  20.]]]
    b1: [6]
    [ 0.  0.  0.  0.  0.  0.]
    et1 [2 3 6]
    [[[ 13.  13.  13.  13.  13.  13.]
      [ 20.  20.  20.  20.  20.  20.]
      [ 28.  28.  28.  28.  28.  28.]]

     [[ 40.  40.  40.  40.  40.  40.]
      [ 38.  38.  38.  38.  38.  38.]
      [ 47.  47.  47.  47.  47.  47.]]]
    v:  [6]
    et: [2 3]
    [[  78.  120.  168.]
     [ 240.  228.  282.]]

    '''