import tensorflow as tf


def noise_shape(tensor, shape):
    output_shape = list()
    tensor_shape = tf.shape(tensor)
    for i, noise_dim in enumerate(shape):
        if noise_dim is None:
            output_shape.append(tensor_shape[i])
        else:
            output_shape.append(noise_dim)
    return output_shape
