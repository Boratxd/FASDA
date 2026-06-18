import tensorflow as tf
from tensorflow.keras import layers

class L1DistanceLayer(layers.Layer):
    def call(self, inputs):
        x, y = inputs
        return tf.abs(x - y)
