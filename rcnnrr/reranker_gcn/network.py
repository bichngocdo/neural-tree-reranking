import numpy as np
import tensorflow as tf

import rcnnrr.nn
from rcnnrr.gcn.gcn import GraphConvLayer
from rcnnrr.reranker_gcn.char_emb import character_based_embeddings
from rcnnrr.nn.tree_ranking_loss import tree_ranking_loss_all, tree_ranking_loss_max


class GCNReranker(object):
    def __init__(self, args):
        self._build_embeddings(args)

        self.train = self._build_train_function(args)
        self.eval = self._build_eval_function(args)

        self.make_train_summary = self._build_train_summary_function()
        self.make_dev_summary = self._build_dev_summary_function()

    def initialize_global_variables(self, session):
        feed_dict = dict()
        if self.word_pt_embeddings is not None:
            feed_dict[self.word_pt_embeddings_ph] = self._word_pt_embeddings
        if self.tag_pt_embeddings is not None:
            feed_dict[self.tag_pt_embeddings_ph] = self._tag_pt_embeddings
        session.run(tf.global_variables_initializer(), feed_dict=feed_dict)

    def _build_train_summary_function(self):
        with tf.variable_scope('train_summary/'):
            x_acc = tf.placeholder(tf.float32,
                                   shape=None,
                                   name='x_acc')
            x_uas = tf.placeholder(tf.float32,
                                   shape=None,
                                   name='x_uas')

            tf.summary.scalar('train_acc', x_acc, collections=['train_summary'])
            tf.summary.scalar('train_uas', x_uas, collections=['train_summary'])

            summary = tf.summary.merge_all(key='train_summary')

        def f(session, acc, uas):
            feed_dict = {
                x_acc: acc,
                x_uas: uas,
            }
            return session.run(summary, feed_dict=feed_dict)

        return f

    def _build_dev_summary_function(self):
        with tf.variable_scope('dev_summary/'):
            x_loss = tf.placeholder(tf.float32,
                                    shape=None,
                                    name='x_loss')
            x_acc = tf.placeholder(tf.float32,
                                   shape=None,
                                   name='x_acc')
            x_uas = tf.placeholder(tf.float32,
                                   shape=None,
                                   name='x_uas')

            tf.summary.scalar('dev_loss', x_loss, collections=['dev_summary'])
            tf.summary.scalar('dev_acc', x_acc, collections=['dev_summary'])
            tf.summary.scalar('dev_uas', x_uas, collections=['dev_summary'])

            summary = tf.summary.merge_all(key='dev_summary')

        def f(session, loss, acc, uas):
            feed_dict = {
                x_loss: loss,
                x_acc: acc,
                x_uas: uas
            }
            return session.run(summary, feed_dict=feed_dict)

        return f

    def _build_placeholders(self):
        # x_word has shape (batch_size, max_length)
        x_word = tf.placeholder(tf.int32,
                                shape=(None, None),
                                name='x_word')
        if self.word_pt_embeddings is not None:
            x_pt_word = tf.placeholder(tf.int32,
                                       shape=(None, None),
                                       name='x_pt_word')
        else:
            x_pt_word = None

        # x_char has shape (batch_size, max_length, max_char_length)
        x_char = tf.placeholder(tf.int32,
                                shape=(None, None, None),
                                name='x_char')

        # x_tag has shape (batch_size, max_length)
        x_tag = tf.placeholder(tf.int32,
                               shape=(None, None),
                               name='x_tag')
        # if self.tag_pt_embeddings is not None:
        #     x_pt_tag = tf.placeholder(tf.int32,
        #                               shape=(None, None),
        #                               name='x_pt_tag')
        # else:
        #     x_pt_tag = None

        # x_edge_in has shape (batch_size, num_trees, max_length, max_degree)
        x_edge_in = tf.placeholder(tf.int32,
                                   shape=(None, None, None, None),
                                   name='x_edge_in')

        # x_label_in has shape (batch_size, num_trees, max_length, max_degree)
        x_label_in = tf.placeholder(tf.int32,
                                    shape=(None, None, None, None),
                                    name='x_label_in')

        # x_edge_out has shape (batch_size, num_trees, max_length, max_degree)
        x_edge_out = tf.placeholder(tf.int32,
                                    shape=(None, None, None, None),
                                    name='x_edge_out')

        # x_label_out has shape (batch_size, num_trees, max_length, max_degree)
        x_label_out = tf.placeholder(tf.int32,
                                     shape=(None, None, None, None),
                                     name='x_label_out')

        # y_label has shape (batch_size, num_trees)
        y_label = tf.placeholder(tf.int32,
                                 shape=(None, None),
                                 name='y_label')

        # y_margin has shape (batch_size, num_trees)
        y_margin = tf.placeholder(tf.float32,
                                  shape=(None, None),
                                  name='y_margin')

        return [x_word, x_pt_word, x_char, x_tag,
                x_edge_in, x_label_in, x_edge_out, x_label_out,
                y_label, y_margin]

    def _build_embeddings(self, args):
        with tf.variable_scope('embeddings'):
            if args.word_embeddings is not None:
                self.word_embeddings = tf.get_variable('word_embeddings',
                                                       shape=(args.no_words, args.word_dim),
                                                       dtype=tf.float32,
                                                       initializer=tf.zeros_initializer,
                                                       regularizer=tf.contrib.layers.l2_regularizer(args.word_l2))
                self.word_pt_embeddings_ph = tf.placeholder(tf.float32,
                                                            shape=args.word_embeddings.shape,
                                                            name='word_pt_embeddings_ph')
                self.word_pt_embeddings = tf.Variable(self.word_pt_embeddings_ph,
                                                      name='word_pt_embeddings',
                                                      trainable=False)
                self._word_pt_embeddings = args.word_embeddings
            else:
                self.word_embeddings = tf.get_variable('word_embeddings',
                                                       shape=(args.no_words, args.word_dim),
                                                       dtype=tf.float32,
                                                       initializer=tf.variance_scaling_initializer(scale=3.,
                                                                                                   distribution='uniform',
                                                                                                   mode='fan_in'),
                                                       regularizer=tf.contrib.layers.l2_regularizer(args.word_l2))
                self.word_pt_embeddings = None

            if args.tag_embeddings is not None:
                self.tag_embeddings = tf.get_variable('tag_embeddings',
                                                      shape=(args.no_tags, args.tag_dim),
                                                      dtype=tf.float32,
                                                      initializer=tf.zeros_initializer,
                                                      regularizer=tf.contrib.layers.l2_regularizer(args.tag_l2))
                self.tag_pt_embeddings_ph = tf.placeholder(tf.float32,
                                                           shape=args.tag_embeddings.shape,
                                                           name='tag_pt_embeddings_ph')
                self.tag_pt_embeddings = tf.Variable(self.tag_pt_embeddings_ph,
                                                     name='tag_pt_embeddings',
                                                     trainable=False)
                self._tag_pt_embeddings = args.tag_embeddings
            else:
                self.tag_embeddings = tf.get_variable('tag_embeddings',
                                                      shape=(args.no_tags, args.tag_dim),
                                                      dtype=tf.float32,
                                                      initializer=tf.variance_scaling_initializer(scale=3.,
                                                                                                  distribution='uniform',
                                                                                                  mode='fan_in'),
                                                      regularizer=tf.contrib.layers.l2_regularizer(args.tag_l2))
                self.tag_pt_embeddings = None

            if args.use_characters:
                self.character_embeddings = tf.get_variable('characters_embeddings',
                                                            shape=(args.no_chars, args.char_dim),
                                                            dtype=tf.float32,
                                                            initializer=tf.variance_scaling_initializer(scale=3.,
                                                                                                        distribution='uniform',
                                                                                                        mode='fan_in'),
                                                            regularizer=tf.contrib.layers.l2_regularizer(
                                                                args.char_l2))

    def _build_input_layers(self, args, is_training):
        def f(x_word, x_pt_word, x_char, x_tag):
            # Word embeddings
            e_word = tf.nn.embedding_lookup(self.word_embeddings, x_word)
            if is_training:
                e_word = tf.nn.dropout(e_word,
                                       keep_prob=1 - args.input_dropout,
                                       noise_shape=rcnnrr.nn.noise_shape(e_word, (None, 1, None)))

            # Word pre-trained embeddings
            if self.word_pt_embeddings is not None:
                e_word_pt = tf.nn.embedding_lookup(self.word_pt_embeddings, x_pt_word)
                if is_training:
                    e_word_pt = tf.nn.dropout(e_word_pt,
                                              keep_prob=1 - args.input_dropout,
                                              noise_shape=rcnnrr.nn.noise_shape(e_word, (None, 1, None)))
                e_word += e_word_pt

            # Character-based word embeddings
            if args.use_characters:
                mask = tf.greater(x_char, 0)
                lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
                e_char = tf.nn.embedding_lookup(self.character_embeddings, x_char)
                e_char_word = character_based_embeddings(e_char,
                                                         args.char_dim,
                                                         args.char_hidden_dim,
                                                         args.word_dim,
                                                         lengths, is_training,
                                                         input_keep_prob=1 - args.char_input_dropout,
                                                         state_keep_prob=1 - args.char_recurrent_dropout,
                                                         output_keep_prob=1 - args.dropout,
                                                         variational_recurrent=True)
                e_word += e_char_word

            # Tag embeddings
            e_tag = tf.nn.embedding_lookup(self.tag_embeddings, x_tag)
            if is_training:
                e_tag = tf.nn.dropout(e_tag,
                                      keep_prob=1 - args.input_dropout,
                                      noise_shape=rcnnrr.nn.noise_shape(e_tag, (None, 1, None)))

            input = tf.concat([e_word, e_tag], axis=-1)
            return input

        return f

    def _build_lstm_layers(self, args, is_training):
        def f(input, lengths):
            hidden = input
            with tf.variable_scope('lstms'):
                for i in range(args.num_lstms):
                    fw_cell = tf.nn.rnn_cell.LSTMCell(args.lstm_hidden_dim)
                    if is_training:
                        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell,
                                                                state_keep_prob=1 - args.recurrent_dropout,
                                                                output_keep_prob=1 - args.dropout,
                                                                variational_recurrent=True,
                                                                dtype=tf.float32)

                    bw_cell = tf.nn.rnn_cell.LSTMCell(args.lstm_hidden_dim)
                    if is_training:
                        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell,
                                                                state_keep_prob=1 - args.recurrent_dropout,
                                                                output_keep_prob=1 - args.dropout,
                                                                variational_recurrent=True,
                                                                dtype=tf.float32)
                    with tf.variable_scope('lstm%d' % i):
                        (fw, bw), (fw_s, bw_s) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, hidden,
                                                                                 sequence_length=lengths,
                                                                                 dtype=tf.float32)
                    hidden = tf.concat([fw, bw], axis=-1)

            with tf.variable_scope('mlp'):
                for _ in range(args.num_mlps):
                    hidden = tf.layers.dense(hidden,
                                             units=args.mlp_dim,
                                             activation=tf.nn.leaky_relu,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(args.l2))

            return hidden

        return f

    def _build_gcn_layers(self, args, is_training):
        def f(inputs, nodes, edges_in, edges_out, labels_in, labels_out):
            hidden = inputs
            input_dim = args.word_dim + args.tag_dim
            if args.num_lstms > 0:
                input_dim = 2 * args.lstm_hidden_dim
            if args.num_mlps > 0:
                input_dim = args.mlp_dim

            with tf.variable_scope('gcns'):
                for i in range(args.num_gcns):
                    if is_training:
                        gcn = GraphConvLayer(input_dim, args.gcn_hidden_dim, args.no_labels,
                                             gated=True, keep_prob=1 - args.gcn_dropout,
                                             weight_regularizer=tf.contrib.layers.l2_regularizer(args.l2))
                    else:
                        gcn = GraphConvLayer(input_dim, args.gcn_hidden_dim, args.no_labels,
                                             gated=True,
                                             weight_regularizer=tf.contrib.layers.l2_regularizer(args.l2))
                    with tf.variable_scope('gcn%d' % i):
                        hidden = gcn(hidden, nodes, edges_in, edges_out, labels_in, labels_out)
                        input_dim = args.gcn_hidden_dim

            return hidden

        return f

    def _build(self, args, is_training):
        with tf.variable_scope('placeholders'):
            x_word, x_pt_word, x_char, x_tag, \
                x_edge_in, x_label_in, x_edge_out, x_label_out, \
                y_label, y_margin = self._build_placeholders()

        input_layers = self._build_input_layers(args, is_training)
        lstm_layers = self._build_lstm_layers(args, is_training)
        gcn_layers = self._build_gcn_layers(args, is_training)

        with tf.variable_scope('input_layers', reuse=tf.AUTO_REUSE):
            inputs = input_layers(x_word, x_pt_word, x_char, x_tag)
            mask = tf.greater(x_word, 0)
            lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)

        with tf.variable_scope('hidden_layers', reuse=tf.AUTO_REUSE):
            hidden = lstm_layers(inputs, lengths)

            batch_size, max_length, _ = tf.unstack(tf.shape(hidden))
            _, num_trees, _, max_in_degree = tf.unstack(tf.shape(x_edge_in))
            _, num_trees, _, max_out_degree = tf.unstack(tf.shape(x_edge_out))

            edges_in = tf.reshape(x_edge_in, (batch_size * num_trees, max_length, max_in_degree))
            edges_out = tf.reshape(x_edge_out, (batch_size * num_trees, max_length, max_out_degree))
            labels_in = tf.reshape(x_label_in, (batch_size * num_trees, max_length, max_in_degree))
            labels_out = tf.reshape(x_label_out, (batch_size * num_trees, max_length, max_out_degree))

            hidden = tf.reshape(tf.tile(hidden, (1, num_trees, 1)),
                                (batch_size * num_trees, max_length, tf.shape(hidden)[-1]))
            nodes = tf.reshape(tf.tile(mask, (1, num_trees)),
                               (batch_size * num_trees, max_length))

            hidden = gcn_layers(hidden, nodes, edges_in, edges_out, labels_in, labels_out)
            hidden = tf.reshape(hidden, (batch_size, num_trees, max_length, args.gcn_hidden_dim))

            with tf.variable_scope('scoring_layer', reuse=tf.AUTO_REUSE):
                # if is_training:
                #     hidden = tf.nn.dropout(hidden, keep_prob=1 - args.dropout,
                #                            noise_shape=rcnnrr.nn.noise_shape(hidden, (None, None, 1, None)))
                scores = tf.layers.dense(hidden, 1,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(args.l2))

        with tf.variable_scope('output_layers', reuse=tf.AUTO_REUSE):
            tree_scores = tf.reduce_sum(scores, axis=[2, 3])
            # tree_scores = scores[:, :, 0, :]
            # tree_scores = tf.squeeze(tree_scores, axis=-1)
            tree_scores = tf.where(tf.greater_equal(y_label, 0),
                                   tree_scores,
                                   tf.ones_like(tree_scores) * np.float('-inf'))

            prediction = tf.argmax(tree_scores, axis=-1)

        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            loss = 0.
            if args.loss == 'ranking_max':
                loss = tree_ranking_loss_max(tree_scores, y_label, y_margin, args.margin_discount)
            elif args.loss == 'ranking_all':
                loss = tree_ranking_loss_all(tree_scores, y_label, y_margin, args.margin_discount)
            else:
                raise Exception('Unknown loss:', args.loss)

        inputs = [x_word, x_pt_word, x_char, x_tag,
                  x_edge_in, x_label_in, x_edge_out, x_label_out,
                  y_label, y_margin]
        outputs = {
            'hidden': hidden,
            'scores': tree_scores,
            'prediction': prediction
        }

        return inputs, outputs, loss

    def _build_train_function(self, args):
        with tf.name_scope('train'):
            inputs, outputs, loss = self._build(args, is_training=True)

            self.iteration = tf.Variable(-1, name='iteration', trainable=False)
            with tf.variable_scope('learning_rate'):
                self.learning_rate = tf.train.exponential_decay(learning_rate=args.learning_rate,
                                                                global_step=self.iteration,
                                                                decay_steps=args.decay_step,
                                                                decay_rate=args.decay_rate)

            with tf.variable_scope('train_summary/'):
                tf.summary.scalar('lr', self.learning_rate, collections=['train_instant_summary'])
                tf.summary.scalar('train_loss', loss, collections=['train_instant_summary'])
                summary = tf.summary.merge_all(key='train_instant_summary')

            if args.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif args.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif args.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif args.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            elif args.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            else:
                raise Exception('Unknown optimizer:', args.optimizer)

            gradients_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
            if args.max_norm is not None:
                with tf.variable_scope('max_norm'):
                    gradients = [gv[0] for gv in gradients_vars]
                    gradients, _ = tf.clip_by_global_norm(gradients, args.max_norm)
                    gradients_vars = [(g, gv[1]) for g, gv in zip(gradients, gradients_vars)]

            with tf.variable_scope('optimizer'):
                train_step = optimizer.apply_gradients(gradients_vars, global_step=self.iteration)

        def f(vars, session):
            feed_dict = dict()
            for input, var in zip(inputs, vars):
                if input is not None:
                    feed_dict[input] = var
            _, output_, loss_, summary_ = session.run([train_step, outputs, loss, summary],
                                                      feed_dict=feed_dict)
            return output_, loss_, summary_

        return f

    def _build_eval_function(self, args):
        with tf.name_scope('eval'):
            inputs, outputs, loss = self._build(args, is_training=False)

        def f(vars, session):
            feed_dict = dict()
            for input, var in zip(inputs, vars):
                if input is not None:
                    feed_dict[input] = var
            return session.run([outputs, loss], feed_dict=feed_dict)

        return f
