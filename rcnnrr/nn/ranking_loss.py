import numpy as np
import tensorflow as tf


def _compute_valid_mask(labels):
    negative_mask = tf.expand_dims(tf.equal(labels, 0), -1)
    positive_mask = tf.expand_dims(tf.equal(labels, 1), -2)
    valid_mask = tf.logical_and(negative_mask, positive_mask)
    return valid_mask


def _compute_valid_mask_orders(orders):
    negative_orders = tf.expand_dims(orders, -1)
    positive_orders = tf.expand_dims(orders, -2)
    valid_mask = tf.greater(negative_orders, positive_orders)
    return valid_mask


def _pairwise_loss(positive_scores, negative_scores, margin=1.):
    return negative_scores - positive_scores + margin


def _logistic_pairwise_loss(positive_scores, negative_scores, positive_margin=2.5, negative_margin=0.5, gamma=2.):
    """
    dos Santos et al., 2015. Classifying Relations by Ranking with Convolutional Neural Networks

    """
    positive_losses = tf.log(1 + tf.exp(gamma * (positive_margin - positive_scores)))
    negative_losses = tf.log(1 + tf.exp(gamma * (negative_margin + negative_scores)))

    return positive_losses + negative_losses


def _ranking_loss_all(scores, labels, loss_func, *loss_args):
    negative_scores = tf.expand_dims(scores, -1)
    positive_scores = tf.expand_dims(scores, -2)
    all_dists = loss_func(positive_scores, negative_scores, *loss_args)

    valid_mask = _compute_valid_mask(labels)
    valid_dists = tf.boolean_mask(all_dists, valid_mask)

    not_easy_mask = tf.greater(valid_dists, 0.)
    not_easy_dists = tf.boolean_mask(valid_dists, not_easy_mask)

    no_pairs = tf.reduce_sum(tf.to_float(not_easy_mask))
    loss = tf.reduce_sum(not_easy_dists) / no_pairs
    loss = tf.cond(tf.equal(no_pairs, 0), lambda: 0., lambda: loss)

    return loss


def _ranking_loss_max(scores, labels, loss_func, *loss_args):
    positive_mask = tf.equal(labels, 1)
    negative_mask = tf.equal(labels, 0)

    positive_scores = tf.where(positive_mask,
                               scores,
                               tf.ones_like(scores) * np.inf)
    positive_scores = tf.reduce_min(positive_scores, -1)

    negative_scores = tf.where(negative_mask,
                               scores,
                               tf.ones_like(scores) * -np.inf)
    negative_scores = tf.reduce_max(negative_scores, -1)

    losses = loss_func(positive_scores, negative_scores, *loss_args)

    not_easy_mask = tf.greater(losses, 0.)
    not_easy_dists = tf.boolean_mask(losses, not_easy_mask)

    no_pairs = tf.reduce_sum(tf.to_float(not_easy_mask))
    loss = tf.reduce_sum(not_easy_dists) / no_pairs
    loss = tf.cond(tf.equal(no_pairs, 0), lambda: 0., lambda: loss)

    return loss


def _ranking_loss_all_orders(scores, orders, loss_func, *loss_args):
    negative_scores = tf.expand_dims(scores, -1)
    positive_scores = tf.expand_dims(scores, -2)
    all_dists = loss_func(positive_scores, negative_scores, *loss_args)

    valid_mask = _compute_valid_mask_orders(orders)
    valid_dists = tf.boolean_mask(all_dists, valid_mask)

    not_easy_mask = tf.greater(valid_dists, 0.)
    not_easy_dists = tf.boolean_mask(valid_dists, not_easy_mask)

    no_pairs = tf.reduce_sum(tf.to_float(not_easy_mask))
    loss = tf.reduce_sum(not_easy_dists) / no_pairs
    loss = tf.cond(tf.equal(no_pairs, 0), lambda: 0., lambda: loss)

    return all_dists, loss


def ranking_loss_all(scores, labels, margin=1.):
    return _ranking_loss_all(scores, labels, _pairwise_loss, margin)


def ranking_loss_max(scores, labels, margin=1.):
    return _ranking_loss_max(scores, labels, _pairwise_loss, margin)


def ranking_loss_all_orders(scores, orders, margin=1.):
    return _ranking_loss_all_orders(scores, orders, _pairwise_loss, margin)


def logisic_ranking_loss_all(scores, labels, positive_margin=2.5, negative_margin=0.5, gamma=2.):
    """
    dos Santos et al., 2015. Classifying Relations by Ranking with Convolutional Neural Networks

    """
    return _ranking_loss_all(scores, labels, _logistic_pairwise_loss, positive_margin, negative_margin, gamma)


def logisic_ranking_loss_max(scores, labels, positive_margin=2.5, negative_margin=0.5, gamma=2.):
    """
    dos Santos et al., 2015. Classifying Relations by Ranking with Convolutional Neural Networks

    """
    return _ranking_loss_max(scores, labels, _logistic_pairwise_loss, positive_margin, negative_margin, gamma)
