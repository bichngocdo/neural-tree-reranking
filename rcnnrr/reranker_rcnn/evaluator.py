from collections import OrderedDict


class Stats(object):
    def __init__(self, name):
        self.name = name

        self.loss = 0.
        self.time = 0.
        self.num_iterations = 0
        self.num_sentences = 0
        self.num_tokens = 0
        self.num_correct_sentences = 0
        self.num_correct_heads = 0

    def reset(self):
        self.loss = 0.
        self.time = 0.
        self.num_iterations = 0
        self.num_sentences = 0
        self.num_tokens = 0
        self.num_correct_sentences = 0
        self.num_correct_heads = 0

    def update(self, loss, time_elapsed, gold_trees, trees, prediction):
        self.loss += loss
        self.time += time_elapsed
        self.num_iterations += 1

        for gold_tree, tree_list, predicted_tree_id in zip(gold_trees, trees, prediction):
            gold_tree = [h for h in gold_tree if h >= 0]
            predicted_tree = [h for h in tree_list[predicted_tree_id] if h >= 0]
            self.num_sentences += 1
            num_errors = 0
            for gold_head, predicted_head in zip(gold_tree, predicted_tree):
                if gold_head >= 0:
                    self.num_tokens += 1
                    self.num_correct_heads += gold_head == predicted_head
                    num_errors += gold_head != predicted_head
            self.num_correct_sentences += num_errors == 0

    def aggregate(self):
        results = OrderedDict()
        results['%s_loss' % self.name] = self.loss / self.num_iterations
        results['%s_rate' % self.name] = self.num_sentences / self.time
        results['%s_acc' % self.name] = self.num_correct_sentences / self.num_sentences \
            if self.num_sentences > 0 else float('NaN')
        results['%s_uas' % self.name] = self.num_correct_heads / self.num_tokens \
            if self.num_tokens > 0 else float('NaN')
        # results['%s_num_tokens' % self.name] = self.num_tokens
        # results['%s_num_correct_heads' % self.name] = self.num_correct_heads
        # results['%s_num_sentences' % self.name] = self.num_sentences
        # results['%s_num_correct_sentences' % self.name] = self.num_correct_sentences
        self.reset()
        return results
