import argparse
import json

PUNCTS = set('!"#%&\'()*+,-.\\/:;<>?@[\]^_{|}~')


def is_punct(word):
    if word in PUNCTS:
        return True
    for c in word:
        if c not in PUNCTS:
            return False
    return True


def analyze(fp, punct=True, k=None):
    best_correct_heads = 0
    worst_correct_heads = 0
    best_correct_labels = 0
    worst_correct_labels = 0
    total_tokens = 0

    one_best_correct_heads = 0
    one_best_correct_labels = 0
    all_correct_heads = 0
    all_correct_labels = 0
    all_tokens = 0

    with open(fp, 'r') as f:
        for line in f:
            data = json.loads(line.strip())

            sentence_length = len([w for w in data['words'] if punct or not is_punct(w)])
            best_tree = 0
            best_labeled_tree = 0
            worst_tree = sentence_length
            worst_labeled_tree = sentence_length
            total_tokens += sentence_length

            best_score = float('-inf')
            best_id = 0

            if k is not None:
                data['nbest_parses'] = data['nbest_parses'][:k]
            for i, tree in enumerate(data['nbest_parses']):
                num_correct_heads = 0
                num_correct_labels = 0
                for word, head, pred_head, label, pred_label in zip(data['words'], data['heads'], tree['heads'],
                                                                    data['labels'], tree['labels']):
                    if punct or not is_punct(word):
                        if head == pred_head:
                            num_correct_heads += 1
                            if label == pred_label:
                                num_correct_labels += 1

                if best_score < tree['score']:
                    best_score = tree['score']
                    best_id = i
                all_correct_heads += num_correct_heads
                all_correct_labels += num_correct_labels
                all_tokens += sentence_length

                if num_correct_heads > best_tree:
                    best_tree = num_correct_heads
                if num_correct_heads < worst_tree:
                    worst_tree = num_correct_heads
                if num_correct_labels > best_labeled_tree:
                    best_labeled_tree = num_correct_labels
                if num_correct_labels < worst_labeled_tree:
                    worst_labeled_tree = num_correct_labels

            num_correct_heads = 0
            num_correct_labels = 0
            tree = data['nbest_parses'][best_id]
            for word, head, pred_head, label, pred_label in zip(data['words'], data['heads'], tree['heads'],
                                                                data['labels'], tree['labels']):
                if punct or word not in PUNCTS:
                    if head == pred_head:
                        num_correct_heads += 1
                        if label == pred_label:
                            num_correct_labels += 1
            one_best_correct_heads += num_correct_heads
            one_best_correct_labels += num_correct_labels

            best_correct_heads += best_tree
            worst_correct_heads += worst_tree
            best_correct_labels += best_labeled_tree
            worst_correct_labels += worst_labeled_tree

        best_uas = 100. * best_correct_heads / total_tokens
        worst_uas = 100. * worst_correct_heads / total_tokens
        best_las = 100. * best_correct_labels / total_tokens
        worst_las = 100. * worst_correct_labels / total_tokens

        one_best_uas = 100. * one_best_correct_heads / total_tokens
        one_best_las = 100. * one_best_correct_labels / total_tokens

        avg_uas = 100. * all_correct_heads / all_tokens
        avg_las = 100. * all_correct_labels / all_tokens

        print('Best UAS : %d / %d = %5.2f' % (best_correct_heads, total_tokens, best_uas))
        print('Worst UAS: %d / %d = %5.2f' % (worst_correct_heads, total_tokens, worst_uas))
        print('Best LAS : %d / %d = %5.2f' % (best_correct_labels, total_tokens, best_las))
        print('Worst LAS: %d / %d = %5.2f' % (worst_correct_labels, total_tokens, worst_las))

        print('1-Best UAS: %d / %d = %5.2f' % (one_best_correct_heads, total_tokens, one_best_uas))
        print('1-Best LAS: %d / %d = %5.2f' % (one_best_correct_labels, total_tokens, one_best_las))

        print('Avg. UAS: %d / %d = %5.2f' % (all_correct_heads, all_tokens, avg_uas))
        print('Avg. LAS: %d / %d = %5.2f' % (all_correct_labels, all_tokens, avg_las))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyzer k-best trees')
    parser.add_argument('input', type=str,
                        help='Input file in JSONL format')
    parser.add_argument('-k', type=int, default=None,
                        help='Analyze top k trees in the list')
    parser.add_argument('-p', action='store_false',
                        help='Analyze without punctuations')
    args = parser.parse_args()

    analyze(args.input, args.p, args.k)
