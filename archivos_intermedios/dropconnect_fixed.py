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
        batch_size = tf.shape(inputs)[0]
        rank = tf.rank(inputs)
        def dropconnect_4d():
            random_tensor = keep_prob + tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            return tf.math.divide(inputs, keep_prob) * binary_tensor
        def dropconnect_2d():
            random_tensor = keep_prob + tf.random.uniform([batch_size, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            return tf.math.divide(inputs, keep_prob) * binary_tensor
        def dropconnect_other():
            random_tensor = keep_prob + tf.random.uniform([batch_size] + [1]*(rank-1), dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            return tf.math.divide(inputs, keep_prob) * binary_tensor
        # Use tf.cond to handle symbolic tensor rank comparison
        output = tf.cond(
            tf.equal(rank, 2),
            true_fn=dropconnect_2d,
            false_fn=lambda: tf.cond(
                tf.equal(rank, 4),
                true_fn=dropconnect_4d,
                false_fn=dropconnect_other
            )
        )
        return output
