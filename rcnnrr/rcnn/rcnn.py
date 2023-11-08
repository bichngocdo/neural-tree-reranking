import numpy as np
import tensorflow as tf


class RCNNCell(object):
    def __init__(self, dim, distance_dim, max_sentence_length, no_tags,
                 regularizer=None):
        self.dim = dim
        self.distance_dim = distance_dim
        self.max_length = max_sentence_length
        self.no_tags = no_tags

        with tf.variable_scope('rcnn_cell'):
            self.d = tf.get_variable('d',
                                     shape=(2 * max_sentence_length + 3, distance_dim),
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                     regularizer=regularizer)
            self.W = tf.get_variable('W',
                                     shape=(no_tags, no_tags, 2 * dim + distance_dim, dim),
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                     regularizer=regularizer)
            self.v = tf.get_variable('v',
                                     shape=(no_tags, no_tags, dim),
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                     regularizer=regularizer)

    def __call__(self, word_embs, phrase_embs, tag_ids, head, children):
        """
        RCNN cell

        :param word_embs: tensor of shape (batch_size, max_length, dim)
        :param phrase_embs: tensor of shape (batch_size, max_length, dim)
        :param tag_ids: tensor of shape (batch_size, max_length)
        :param head: tensor of shape (batch_size,)
        :param children: tensor of shape (batch_size, num_children)
        :return: phrase_embs, score
        """
        with tf.variable_scope('rcnn_cell'):
            max_length = tf.shape(word_embs)[1]

            # Masking head with id -1, then replace with 0
            head_mask = tf.greater_equal(head, 0)
            head = tf.where(head_mask, head, tf.zeros_like(head))

            # Output mask for phrase embeddings
            output_mask_ = tf.expand_dims(tf.one_hot(head, max_length), -1)
            output_mask = tf.where(head_mask, output_mask_, tf.zeros_like(output_mask_))
            output_mask = tf.greater(output_mask, 0)
            output_mask = tf.tile(output_mask, (1, 1, self.dim))

            head = tf.tile(tf.expand_dims(head, -1), (1, tf.shape(children)[1]))
            head_tag = tf.batch_gather(tag_ids, head, name='batch_gather')
            child_tags = tf.batch_gather(tag_ids, children, name='batch_gather')
            head_child_tag = tf.stack([head_tag, child_tags], axis=-1)
            distance = head - children + self.max_length + 1
            # Out of range distances
            distance = tf.where(tf.less(distance, 0),
                                distance,
                                tf.zeros_like(distance))
            distance = tf.where(tf.greater(distance, 2 * self.max_length + 2),
                                distance,
                                tf.ones_like(distance) * (2 * self.max_length + 2))

            x_word_head = tf.batch_gather(word_embs, head, name='batch_gather')
            x_phrase_children = tf.batch_gather(phrase_embs, children, name='batch_gather')
            x_distance = tf.gather(self.d, distance)
            p = tf.concat([x_word_head, x_phrase_children, x_distance], axis=-1)
            p = tf.expand_dims(p, 2)  # p has shape (batch_size, num_children, 1, 3 * dim)

            W = tf.gather_nd(self.W, head_child_tag)  # W has shape (batch_size, num_children, 3 * dim, dim)
            v = tf.gather_nd(self.v, head_child_tag)  # v has shape (batch_size, num_children, dim)

            z = tf.squeeze(tf.tanh(tf.matmul(p, W)), axis=2)  # z has shape (batch_size, num_children, dim)

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


class RCNNDropoutWrapper(object):
    def __init__(self, cell, keep_prob=1.0):
        self.cell = cell
        self.keep_prob = keep_prob

    def __call__(self, word_embs, phrase_embs, tag_ids, head, children):
        if self.keep_prob < 1.:
            phrase_embs = tf.nn.dropout(phrase_embs, keep_prob=self.keep_prob)
        return self.cell(word_embs, phrase_embs, tag_ids, head, children)


def _transpose(x):
    dims = tf.range(2, tf.rank(x))
    dims = tf.concat([[1, 0], dims], axis=-1)
    return tf.transpose(x, dims)


def rcnn(cell, inputs, tags, heads, children):
    """
    RCNN on trees

    :param cell: RCNN cell
    :param inputs: input (word) embeddings, tensor of shape [batch_size, max_length, dim]
    :param tags: tensor of shape [batch_size, max_length]
    :param heads: tensor of shape [batch_size, num_heads]
    :param children: tensor of shape [batch_size, num_heads, num_children]
    :return: outputs, score

      outputs: output (phrase) embeddings, tensor of shape [batch_size, max_length, dim]
      score: tree score, tensor of shape [batch_size,]
    """

    # Transpose input tensors
    heads = _transpose(heads)
    children = _transpose(children)

    def step(prev_outputs, step_inputs):
        prev_phrase_embeddings, prev_scores = prev_outputs
        step_head, step_children = step_inputs
        step_phrase_embeddings, step_scores = cell(inputs, prev_phrase_embeddings,
                                                   tags, step_head, step_children)
        return step_phrase_embeddings, step_scores

    phrase_embeddings, scores = tf.scan(step, [heads, children],
                                        initializer=(inputs, tf.zeros(tf.shape(heads)[1])))
    return phrase_embeddings[-1], _transpose(scores)
