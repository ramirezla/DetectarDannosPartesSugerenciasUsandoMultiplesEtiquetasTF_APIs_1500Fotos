from tensorflow.keras import layers
import tensorflow as tf

class DropConnect(layers.Layer):
    def __init__(self, drop_prob=0.5, **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.drop_prob = drop_prob
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        keep_prob = 1.0 - self.drop_prob
        # Apply dropout mask scaled by keep_prob
        # noise_shape same as inputs shape except batch dimension
        noise_shape = tf.shape(inputs)
        # For 2D inputs, noise_shape is [batch_size, 1]
        if len(inputs.shape) == 2:
            noise_shape = [tf.shape(inputs)[0], 1]
        else:
            noise_shape = tf.shape(inputs)
        return tf.nn.dropout(inputs, rate=self.drop_prob, noise_shape=noise_shape) / keep_prob
