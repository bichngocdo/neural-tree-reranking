import tensorflow as tf


def _transpose(x):
    dims = tf.range(2, tf.rank(x))
    dims = tf.concat([[1, 0], dims], axis=-1)
    return tf.transpose(x, dims)


def dynamic_rnn(cell, inputs, lengths=None,
                input_keep_prob=1., output_keep_prob=1.,
                variational_recurrent=False):
    with tf.variable_scope('dynamic_rnn'):
        inputs = _transpose(inputs)

        max_length, batch_size, input_dim = tf.unstack(tf.shape(inputs))
        output_dim = cell.output_dim
        time = tf.constant(0, name='time', dtype=tf.int32)

        if 0 <= input_keep_prob < 1:
            if variational_recurrent:
                noise_shape = (1, batch_size, input_dim)
            else:
                noise_shape = None
            inputs = tf.nn.dropout(inputs, input_keep_prob, noise_shape=noise_shape)

        state = cell.zero_state(batch_size)
        zero_output = state[0]
        outputs = tf.TensorArray(inputs.dtype, size=max_length, name='dynamic_rnn_outputs')

        def step(time, state, outputs):
            input = inputs[time]
            output, new_state = cell(input, state)

            if lengths is not None:
                output = tf.where(tf.greater_equal(time, lengths), zero_output, output)
                if isinstance(state, tuple):
                    new_states = list()
                    for s, ns in zip(state, new_state):
                        new_states.append(tf.where(tf.greater_equal(time, lengths), s, ns))
                    new_state = tuple(new_states)
                else:
                    new_state = tf.where(tf.greater_equal(time, lengths), state, new_state)

            outputs = outputs.write(time, output)
            return time + 1, new_state, outputs

        _, states, outputs = tf.while_loop(cond=lambda time, _1, _2: tf.less(time, max_length),
                                           body=lambda time, state, outputs: step(time, state, outputs),
                                           loop_vars=[time, state, outputs])
        outputs = outputs.stack()

        if 0 < output_keep_prob < 1:
            if variational_recurrent:
                noise_shape = (1, batch_size, output_dim)
            else:
                noise_shape = None
            outputs = tf.nn.dropout(outputs, output_keep_prob, noise_shape=noise_shape)

        outputs = _transpose(outputs)

        return outputs, states


def bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, lengths=None,
                              input_keep_prob=1., output_keep_prob=1.,
                              variational_recurrent=False):
    with tf.variable_scope('bidirectional_dynamic_rnn'):
        with tf.variable_scope('fw'):
            fw_outputs, fw_states = dynamic_rnn(fw_cell, inputs, lengths,
                                                input_keep_prob, output_keep_prob,
                                                variational_recurrent)
        with tf.variable_scope('bw'):
            if lengths is None:
                batch_size, max_length, _ = tf.unstack(tf.shape(inputs))
                seq_lengths = tf.ones(batch_size, dtype=tf.int32) * max_length
            else:
                seq_lengths = lengths
            inputs = tf.reverse_sequence(inputs, seq_lengths, seq_axis=1, batch_axis=0)
            bw_outputs, bw_states = dynamic_rnn(bw_cell, inputs, lengths,
                                                input_keep_prob, output_keep_prob,
                                                variational_recurrent)
            bw_outputs = tf.reverse_sequence(bw_outputs, seq_lengths, seq_axis=1, batch_axis=0)
        return (fw_outputs, bw_outputs), (fw_states, bw_states)
