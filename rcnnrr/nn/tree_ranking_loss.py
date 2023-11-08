import numpy as np
import tensorflow as tf


def _compute_valid_mask(labels):
    negative_mask = tf.expand_dims(tf.equal(labels, 0), -1)
    positive_mask = tf.expand_dims(tf.equal(labels, 1), -2)
    valid_mask = tf.logical_and(negative_mask, positive_mask)
    return valid_mask


def _compute_valid_mask_orders(orders):
    negative_mask = tf.expand_dims(tf.greater_equal(orders, 0), -1)
    positive_mask = tf.expand_dims(tf.greater_equal(orders, 0), -2)
    valid_mask = tf.logical_and(negative_mask, positive_mask)

    negative_orders = tf.expand_dims(orders, -1)
    positive_orders = tf.expand_dims(orders, -2)
    valid_mask = tf.logical_and(valid_mask,
                                tf.greater(negative_orders, positive_orders))

    return valid_mask


def tree_ranking_loss_all(scores, labels, margins, alpha=1.):
    """
    Pairwise ranking loss with structured margin loss
    on all (gold, predicted) pairs of tree

    :param scores: tensor of shape (..., num_trees), ndim >= 2
    :param labels: tensor of shape (..., num_trees), ndim >= 2
    :param margins: tensor of shape (..., num_trees), ndim >= 2
    :return: loss
    """
    negative_scores = tf.expand_dims(scores, -1)
    positive_scores = tf.expand_dims(scores, -2)
    negative_margins = tf.expand_dims(margins, -1)
    positive_margins = tf.expand_dims(margins, -2)
    all_dists = negative_scores - positive_scores + alpha * (negative_margins - positive_margins)

    valid_mask = _compute_valid_mask(labels)
    valid_dists = tf.boolean_mask(all_dists, valid_mask)

    not_easy_mask = tf.greater(valid_dists, 0.)
    not_easy_dists = tf.boolean_mask(valid_dists, not_easy_mask)

    no_pairs = tf.reduce_sum(tf.to_float(not_easy_mask))
    loss = tf.reduce_sum(not_easy_dists) / no_pairs
    loss = tf.cond(tf.equal(no_pairs, 0), lambda: 0., lambda: loss)

    return loss


def tree_ranking_loss_all_orders(scores, orders, margins, alpha=1.):
    """
    Pairwise ranking loss with structured margin loss
    on all (gold, predicted) pairs of tree

    :param scores: tensor of shape (..., num_trees), ndim >= 2
    :param orders: tensor of shape (..., num_trees), ndim >= 2
    :param margins: tensor of shape (..., num_trees), ndim >= 2
    :return: loss
    """
    negative_scores = tf.expand_dims(scores, -1)
    positive_scores = tf.expand_dims(scores, -2)
    negative_margins = tf.expand_dims(margins, -1)
    positive_margins = tf.expand_dims(margins, -2)
    all_dists = negative_scores - positive_scores + alpha * (negative_margins - positive_margins)

    valid_mask = _compute_valid_mask_orders(orders)
    valid_dists = tf.boolean_mask(all_dists, valid_mask)

    not_easy_mask = tf.greater(valid_dists, 0.)
    not_easy_dists = tf.boolean_mask(valid_dists, not_easy_mask)

    no_pairs = tf.reduce_sum(tf.to_float(not_easy_mask))
    loss = tf.reduce_sum(not_easy_dists) / no_pairs
    loss = tf.cond(tf.equal(no_pairs, 0), lambda: 0., lambda: loss)

    return loss


def tree_ranking_loss_max(scores, labels, margins, alpha=1.):
    """
    Pairwise ranking loss with structured margin loss
    between the predicted tree with the highest score
    and the gold tree

    :param scores: tensor of shape (..., num_trees), ndim >= 2
    :param labels: tensor of shape (..., num_trees), ndim >= 2
    :param margins: tensor of shape (..., num_trees), ndim >= 2
    :return: loss
    """
    positive_mask = tf.equal(labels, 1)
    negative_mask = tf.equal(labels, 0)

    positive_scores = tf.where(positive_mask,
                               scores,
                               tf.ones_like(scores) * np.inf)
    positive_indexes = tf.expand_dims(tf.argmin(positive_scores, -1, output_type=tf.int32), -1)
    positive_scores = tf.batch_gather(positive_scores, positive_indexes, name='batch_gather')
    positive_margins = tf.batch_gather(margins, positive_indexes, name='batch_gather')

    negative_scores = tf.where(negative_mask,
                               scores,
                               tf.ones_like(scores) * -np.inf)
    negative_indexes = tf.expand_dims(tf.argmax(negative_scores, -1, output_type=tf.int32), -1)
    negative_scores = tf.batch_gather(negative_scores, negative_indexes, name='batch_gather')
    negative_margins = tf.batch_gather(margins, negative_indexes, name='batch_gather')

    losses = negative_scores - positive_scores + alpha * (negative_margins - positive_margins)

    not_easy_mask = tf.greater(losses, 0.)
    not_easy_dists = tf.boolean_mask(losses, not_easy_mask)

    no_pairs = tf.reduce_sum(tf.to_float(not_easy_mask))
    loss = tf.reduce_sum(not_easy_dists) / no_pairs
    loss = tf.cond(tf.equal(no_pairs, 0), lambda: 0., lambda: loss)

    return loss
