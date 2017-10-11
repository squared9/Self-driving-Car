import tensorflow as tf


# custom init with the seed set to 0 by default
def custom_init(shape, dtype=tf.float32, partition_info=None, seed=0):
    return tf.random_normal(shape, dtype=dtype, seed=seed)


# Set the `kernel_size` and `stride`.
def conv_1x1(x, num_outputs, kernel_size=1, stride=(1, 1)):
    return tf.layers.conv2d(x, num_outputs, kernel_size, stride, weights_initializer=custom_init)


def upsample(value, num_classes=2, output_shape=4, stride=(2, 2)):
    """
    Apply a two times upsample on x and return the result.
    :value: 4-Rank Tensor (NHWC)
    :return: TF Operation
    """
    return tf.layers.conv2d_transpose(value, num_classes, output_shape, stride)


def skip_connection(input, skip_source, num_classes, num_filters=16, strides=(8, 8)):
    input = tf.add(input, skip_source)
    Input = tf.layers.conv2d_transpose(input, num_classes, num_filters, strides)