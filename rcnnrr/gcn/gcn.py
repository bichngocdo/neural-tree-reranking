import tensorflow as tf


class GraphConvLayer(object):
    def __init__(self, input_dim, output_dim, no_labels, gated=True,
                 keep_prob=1., weight_regularizer=None, bias_regularizer=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.no_labels = no_labels
        self.gated = gated

        self.keep_prob = keep_prob
        self.weight_regularizer = weight_regularizer
        self.bias_regularizer = bias_regularizer

    def __call__(self, inputs, nodes, edges_in, edges_out, labels_in, labels_out):
        """

        :param inputs: tensor of shape (batch_size, max_length, input_dim)
        :param nodes: binary tensor of shape (batch_size, max_length)
        :param edges_in: tensor of shape (batch_size, max_length, max_in_degree)
        :param edges_out: tensor of shape (batch_size, max_length, max_out_degree)
        :param labels_in: tensor of shape (batch_size, max_length, max_in_degree)
        :param labels_out: tensor of shape (batch_size, max_length, max_out_degree)
        :return: tensor of shape (batch_size, max_length, output_dim)
        """
        with tf.variable_scope('gcn'):
            batch_size, max_length, input_dim = tf.unstack(tf.shape(inputs))
            max_in_degree = tf.shape(edges_in)[2]
            max_out_degree = tf.shape(edges_out)[2]

            W_in = tf.get_variable('W_in',
                                   shape=(self.input_dim, self.output_dim),
                                   dtype=tf.float32,
                                   initializer=tf.initializers.glorot_normal(),
                                   regularizer=self.weight_regularizer)
            W_out = tf.get_variable('W_out',
                                    shape=(self.input_dim, self.output_dim),
                                    dtype=tf.float32,
                                    initializer=tf.initializers.glorot_normal(),
                                    regularizer=self.weight_regularizer)
            W_self = tf.get_variable('W_self',
                                     shape=(self.input_dim, self.output_dim),
                                     dtype=tf.float32,
                                     initializer=tf.initializers.glorot_normal(),
                                     regularizer=self.weight_regularizer)
            b_in = tf.get_variable('b_in',
                                   shape=(self.no_labels - 1, self.output_dim),
                                   dtype=tf.float32,
                                   initializer=tf.initializers.zeros,
                                   regularizer=self.bias_regularizer)
            b_in = tf.pad(b_in, [[1, 0], [0, 0]],
                          mode='constant', constant_values=0.)
            b_out = tf.get_variable('b_out',
                                    shape=(self.no_labels - 1, self.output_dim),
                                    dtype=tf.float32,
                                    initializer=tf.initializers.zeros,
                                    regularizer=self.bias_regularizer)
            b_out = tf.pad(b_out, [[1, 0], [0, 0]],
                           mode='constant', constant_values=0.)
            b_self = tf.get_variable('b_self',
                                     shape=(1, self.output_dim),
                                     dtype=tf.float32,
                                     initializer=tf.initializers.zeros,
                                     regularizer=self.bias_regularizer)

            # Compute masks and replace -1 index with 0
            mask_in = tf.greater_equal(edges_in, 0)
            mask_out = tf.greater_equal(edges_out, 0)
            edges_in = tf.where(mask_in, edges_in, tf.zeros_like(edges_in))
            edges_out = tf.where(mask_out, edges_out, tf.zeros_like(edges_out))

            # Reshape
            edges_in = tf.reshape(edges_in, (batch_size, max_length * max_in_degree))
            edges_out = tf.reshape(edges_out, (batch_size, max_length * max_out_degree))
            labels_in = tf.reshape(labels_in, (batch_size, max_length * max_in_degree))
            labels_out = tf.reshape(labels_out, (batch_size, max_length * max_out_degree))

            # In edges
            inputs_in = tf.tensordot(inputs, W_in, axes=[[2], [0]])  # (b x t x h)
            inputs_in = tf.batch_gather(inputs_in, edges_in, name='batch_gather')  # (b x t x d * h)
            inputs_in += tf.gather(b_in, labels_in)
            inputs_in = tf.reshape(inputs_in,
                                   (batch_size, max_length, max_in_degree, self.output_dim))  # (b x t x d x h)
            inputs_in *= tf.to_float(tf.expand_dims(mask_in, -1))

            # Out edges
            inputs_out = tf.tensordot(inputs, W_out, axes=[[2], [0]])
            inputs_out = tf.batch_gather(inputs_out, edges_out, name='batch_gather')
            inputs_out += tf.gather(b_out, labels_out)
            inputs_out = tf.reshape(inputs_out,
                                    (batch_size, max_length, max_out_degree, self.output_dim))
            inputs_out *= tf.to_float(tf.expand_dims(mask_out, -1))

            # Self edges
            inputs_self = tf.tensordot(inputs, W_self, axes=[[2], [0]])
            inputs_self += b_self
            inputs_self *= tf.to_float(tf.expand_dims(nodes, -1))
            inputs_self = tf.reshape(inputs_self,
                                     (batch_size, max_length, 1, self.output_dim))

            if self.gated:
                w_gate_in = tf.get_variable('w_gate_in',
                                            shape=(self.input_dim, 1),
                                            dtype=tf.float32,
                                            initializer=tf.initializers.random_uniform(),
                                            regularizer=self.weight_regularizer)
                w_gate_out = tf.get_variable('w_gate_out',
                                             shape=(self.input_dim, 1),
                                             dtype=tf.float32,
                                             initializer=tf.initializers.random_uniform(),
                                             regularizer=self.weight_regularizer)
                w_gate_self = tf.get_variable('w_gate_self',
                                              shape=(self.input_dim, 1),
                                              dtype=tf.float32,
                                              initializer=tf.initializers.random_uniform(),
                                              regularizer=self.weight_regularizer)
                b_gate_in = tf.get_variable('b_gate_in',
                                            shape=(self.no_labels, 1),
                                            dtype=tf.float32,
                                            initializer=tf.initializers.ones,
                                            regularizer=self.bias_regularizer)
                b_gate_out = tf.get_variable('b_gate_out',
                                             shape=(self.no_labels, 1),
                                             dtype=tf.float32,
                                             initializer=tf.initializers.ones,
                                             regularizer=self.bias_regularizer)
                b_gate_self = tf.get_variable('b_gate_self',
                                              shape=(),
                                              dtype=tf.float32,
                                              initializer=tf.initializers.ones,
                                              regularizer=self.bias_regularizer)

                gate_in = tf.tensordot(inputs, w_gate_in, axes=[[2], [0]])  # (b x t x 1)
                gate_in = tf.batch_gather(gate_in, edges_in, name='batch_gather')  # (b x t x d * 1)
                gate_in += tf.gather(b_gate_in, labels_in)
                gate_in = tf.sigmoid(gate_in)
                gate_in = tf.reshape(gate_in,
                                     (batch_size, max_length, max_in_degree, 1))  # (b x t x d x 1)
                gate_in *= tf.to_float(tf.expand_dims(mask_in, -1))

                gate_out = tf.tensordot(inputs, w_gate_out, axes=[[2], [0]])
                gate_out = tf.batch_gather(gate_out, edges_out, name='batch_gather')
                gate_out += tf.gather(b_gate_out, labels_out)
                gate_out = tf.sigmoid(gate_out)
                gate_out = tf.reshape(gate_out,
                                      (batch_size, max_length, max_out_degree, 1))
                gate_out *= tf.to_float(tf.expand_dims(mask_out, -1))

                gate_self = tf.tensordot(inputs, w_gate_self, axes=[[2], [0]])
                gate_self += b_gate_self
                gate_self = tf.sigmoid(gate_self)
                gate_self *= tf.to_float(tf.expand_dims(nodes, -1))
                gate_self = tf.reshape(gate_self,
                                       (batch_size, max_length, 1, 1))

                inputs_in *= gate_in
                inputs_out *= gate_out
                inputs_self *= gate_self

            outputs = tf.concat([inputs_in, inputs_out, inputs_self], axis=2)
            if 0 <= self.keep_prob < 1:
                outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob,
                                        noise_shape=(batch_size, max_length, max_in_degree + max_out_degree + 1, 1))
            outputs = tf.nn.leaky_relu(tf.reduce_sum(outputs, axis=2))

            return outputs
