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
        self.num_correct_labels = 0

    def reset(self):
        self.loss = 0.
        self.time = 0.
        self.num_iterations = 0
        self.num_sentences = 0
        self.num_tokens = 0
        self.num_correct_sentences = 0
        self.num_correct_heads = 0
        self.num_correct_labels = 0

    def update(self, loss, time_elapsed, gold_heads, all_heads, gold_labels, all_labels, prediction):
        self.loss += loss
        self.time += time_elapsed
        self.num_iterations += 1

        for g_heads, head_list, g_labels, label_list, predicted_tree_id in \
                zip(gold_heads, all_heads, gold_labels, all_labels, prediction):
            g_heads = [h for h in g_heads if h >= 0]
            p_heads = [h for h in head_list[predicted_tree_id] if h >= 0]
            g_labels = [l for l in g_labels if l > 0]
            p_labels = [l for l in label_list[predicted_tree_id] if l > 0]

            self.num_sentences += 1
            num_errors = 0
            for g_head, p_head, g_label, p_label in \
                    zip(g_heads, p_heads, g_labels, p_labels):
                if g_head >= 0:
                    self.num_tokens += 1
                    if g_head == p_head:
                        self.num_correct_heads += 1
                        if g_label == p_label:
                            self.num_correct_labels += 1
                    else:
                        num_errors += 1
            self.num_correct_sentences += num_errors == 0

    def aggregate(self):
        results = OrderedDict()
        results['%s_loss' % self.name] = self.loss / self.num_iterations
        results['%s_rate' % self.name] = self.num_sentences / self.time
        results['%s_acc' % self.name] = self.num_correct_sentences / self.num_sentences \
            if self.num_sentences > 0 else float('NaN')
        results['%s_uas' % self.name] = self.num_correct_heads / self.num_tokens \
            if self.num_tokens > 0 else float('NaN')
        results['%s_las' % self.name] = self.num_correct_labels / self.num_tokens \
            if self.num_tokens > 0 else float('NaN')
        self.reset()
        return results
