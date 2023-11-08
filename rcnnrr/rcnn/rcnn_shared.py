import numpy as np
import tensorflow as tf

import rcnnrr.nn


class RCNNCell(object):
    def __init__(self, dim, distance_dim, max_sentence_length,
                 regularizer=None):
        self.dim = dim
        self.distance_dim = distance_dim
        self.max_length = max_sentence_length
        self.regularizer = regularizer

    def __call__(self, input_embs, phrase_embs, head, children, mask=None):
        """
        RCNN cell

        :param input_embs: tensor of shape (batch_size, max_length, dim)
        :param phrase_embs: tensor of shape (batch_size, max_length, dim)
        :param head: tensor of shape (batch_size,)
        :param children: tensor of shape (batch_size, num_children)
        :return: phrase_embs, score
        """
        with tf.variable_scope('rcnn_cell'):
            d = tf.get_variable('d',
                                shape=(2 * self.max_length + 3, self.distance_dim),
                                dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                regularizer=self.regularizer)
            W = tf.get_variable('W',
                                shape=(2 * self.dim + self.distance_dim, self.dim),
                                dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                regularizer=self.regularizer)
            v = tf.get_variable('v',
                                shape=(self.dim,),
                                dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                regularizer=self.regularizer)

            max_length = tf.shape(input_embs)[1]

            # Masking head with id -1, then replace with 0
            head_mask = tf.greater_equal(head, 0)
            head = tf.where(head_mask, head, tf.zeros_like(head))

            # Output mask for phrase embeddings
            output_mask_ = tf.expand_dims(tf.one_hot(head, max_length), -1)
            output_mask = tf.where(head_mask, output_mask_, tf.zeros_like(output_mask_))
            output_mask = tf.greater(output_mask, 0)
            output_mask = tf.tile(output_mask, (1, 1, self.dim))

            head = tf.tile(tf.expand_dims(head, -1), (1, tf.shape(children)[1]))
            distance = head - children + self.max_length + 1
            # Out of range distances
            distance = tf.where(tf.less(distance, 0),
                                distance,
                                tf.zeros_like(distance))
            distance = tf.where(tf.greater(distance, 2 * self.max_length + 2),
                                distance,
                                tf.ones_like(distance) * (2 * self.max_length + 2))

            # Recursive dropout
            phrase_embs_dropped = phrase_embs * mask if mask is not None else phrase_embs

            x_word_head = tf.batch_gather(input_embs, head, name='batch_gather')
            x_phrase_children = tf.batch_gather(phrase_embs_dropped, children, name='batch_gather')
            x_distance = tf.gather(d, distance)
            p = tf.concat([x_word_head, x_phrase_children, x_distance],
                          axis=-1)  # p has shape (batch_size, num_children, 3 * dim)

            z = tf.tanh(tf.tensordot(p, W, axes=[[2], [0]]))  # z has shape (batch_size, num_children, dim)

            children_mask = tf.tile(tf.expand_dims(tf.greater(children, 0), axis=-1), (1, 1, self.dim))

            # Phrase representation of head
            z_p = tf.where(children_mask, z, tf.ones_like(z) * np.float('-inf'))
            x_phrase_head = tf.reduce_max(z_p, 1)  # x_phrase_head has shape (batch_size, dim)

            # Update phrase embeddings
            x_phrase_head = tf.tile(tf.expand_dims(x_phrase_head, 1), (1, max_length, 1))
            phrase_embs = tf.where(output_mask, x_phrase_head, phrase_embs)  # Note: 0. * inf = nan

            # Head score
            z_s = tf.where(children_mask, z, tf.zeros_like(z))
            score = tf.reduce_sum(z_s * v, axis=[1, 2])
            score = tf.where(head_mask, score, tf.zeros_like(score))

            return phrase_embs, score


def _transpose(x):
    dims = tf.range(2, tf.rank(x))
    dims = tf.concat([[1, 0], dims], axis=-1)
    return tf.transpose(x, dims)


def rcnn(cell, words, heads, children, keep_prob=1.0):
    """
    RCNN on trees

    :param cell: RCNN cell
    :param words: input (word) embeddings, tensor of shape [batch_size, max_length, dim]
    :param heads: tensor of shape [batch_size, num_heads]
    :param children: tensor of shape [batch_size, num_heads, num_children]
    :param keep_prob:
    :return: outputs, score

      outputs: output (phrase) embeddings, tensor of shape [batch_size, max_length, dim]
      score: tree score, tensor of shape [batch_size,]
    """

    # Transpose input tensors
    heads = _transpose(heads)
    children = _transpose(children)

    if 0. < keep_prob < 1.:
        ones = tf.ones_like(words)
        mask = tf.nn.dropout(ones, keep_prob, noise_shape=rcnnrr.nn.noise_shape(ones, (None, 1, None)))
    else:
        mask = 1.

    def step(prev_outputs, step_inputs):
        prev_phrase_embeddings, prev_scores = prev_outputs
        step_head, step_children = step_inputs
        step_phrase_embeddings, step_scores = cell(words, prev_phrase_embeddings,
                                                   step_head, step_children, mask)
        return step_phrase_embeddings, step_scores

    phrase_embeddings, scores = tf.scan(step, [heads, children],
                                        initializer=(words, tf.zeros(tf.shape(heads)[1])))
    return phrase_embeddings[-1], _transpose(scores)
