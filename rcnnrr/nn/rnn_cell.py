from abc import ABC, abstractmethod

import tensorflow as tf


class RNNCell(ABC):
    @abstractmethod
    def zero_state(self, batch_size):
        pass

    @abstractmethod
    def state_dropout(self, state, mask):
        pass

    @abstractmethod
    def __call__(self, input_t, state_prev_t):
        return None, None


class LSTMCell(RNNCell):
    def __init__(self, input_dim, output_dim,
                 kernel_regularizer=None, bias_regularizer=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def zero_state(self, batch_size):
        return tf.zeros((batch_size, self.output_dim), name='hidden_state_zero'), \
            tf.zeros((batch_size, self.output_dim), name='cell_state_zero')

    def state_dropout(self, state, mask):
        hidden_state, cell_state = state
        return hidden_state * mask, cell_state

    def __call__(self, input_t, state_prev_t):
        """

        :param input: tensor of shape (batch_size, input_dim)
        :param state: cell and hidden tensors of shape (batch_size, hidden_dim)
        :return: tuple output, new_state (cell, hidden)
        """
        with tf.variable_scope('lstm_cell'):
            hidden_prev_t, cell_prev_t = state_prev_t
            lstm_input = tf.concat([input_t, hidden_prev_t], axis=-1)
            lstm_output = tf.layers.dense(lstm_input,
                                          units=4 * self.output_dim,
                                          use_bias=True,
                                          activation=None,
                                          kernel_regularizer=self.kernel_regularizer,
                                          bias_regularizer=self.bias_regularizer)
            cell_tilde_t, input_t, output_t, forget_t = tf.split(lstm_output, 4, axis=-1)
            cell_tilde_t = tf.tanh(cell_tilde_t)
            input_t = tf.sigmoid(input_t)
            forget_t = tf.sigmoid(forget_t + 1.)
            output_t = tf.sigmoid(output_t)
            cell_t = input_t * cell_tilde_t + forget_t * cell_prev_t
            hidden_t = output_t * tf.tanh(cell_t)
            return hidden_t, (hidden_t, cell_t)


class DummyCell(RNNCell):
    def __init__(self, dim):
        self.input_dim = dim
        self.output_dim = dim

    def zero_state(self, batch_size):
        return tf.zeros((batch_size, self.output_dim), name='state_zero')

    def state_dropout(self, state, mask):
        return state

    def __call__(self, input_t, state_prev_t):
        return input_t, state_prev_t


def dropout_mask(shape, keep_prob):
    if 0 < keep_prob < 1:
        mask = tf.random_uniform(shape, 0, 1)
        mask += keep_prob
        mask = tf.floor(mask) / keep_prob
        return mask
    else:
        return 1.


class DropoutWrapper(RNNCell):
    def __init__(self, cell: RNNCell,
                 input_keep_prob=1.,
                 output_keep_prob=1.,
                 state_keep_prob=1.,
                 variational_recurrent=False):
        self.cell = cell
        self.input_dim = cell.input_dim
        self.output_dim = cell.output_dim
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        self.state_keep_prob = state_keep_prob
        self.variational_recurrent = variational_recurrent

    def __dropout_mask(self, batch_size, dim, keep_prob):
        input_mask = dropout_mask((batch_size, dim), keep_prob)

        def f():
            if self.variational_recurrent:
                return input_mask
            else:
                return dropout_mask((batch_size, dim), keep_prob)

        return f

    def initialize_masks(self, batch_size):
        input_mask = dropout_mask((batch_size, self.input_dim), self.input_keep_prob)
        output_mask = dropout_mask((batch_size, self.output_dim), self.output_keep_prob)
        state_mask = dropout_mask((batch_size, self.output_dim), self.state_keep_prob)
        return input_mask, output_mask, state_mask

    def zero_state(self, batch_size):
        masks = self.initialize_masks(batch_size)
        state = self.cell.zero_state(batch_size)
        if not isinstance(state, tuple):
            state = (state,)
        results = list()
        results.extend(state)
        results.extend(masks)
        return tuple(results)

    def state_dropout(self, state, mask):
        return state

    def __call__(self, input_t, state_prev_t):
        input_mask_prev_t = state_prev_t[-3]
        output_mask_prev_t = state_prev_t[-2]
        state_mask_prev_t = state_prev_t[-1]
        state_prev_t = state_prev_t[:-3]

        if self.variational_recurrent:
            input_mask_t = input_mask_prev_t
            output_mask_t = output_mask_prev_t
            state_mask_t = state_mask_prev_t
        else:
            input_mask_t, output_mask_t, state_mask_t = self.initialize_masks(tf.shape(input_t)[0])

        input_t *= input_mask_t
        state_prev_t = self.cell.state_dropout(state_prev_t, state_mask_t)
        output, state = self.cell(input_t, state_prev_t)
        output *= output_mask_t

        if not isinstance(state, tuple):
            state = (state,)

        return output, state + (input_mask_t, output_mask_t, state_mask_t)
