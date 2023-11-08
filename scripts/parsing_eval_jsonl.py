import argparse
import json

PUNCTS = '-!"#%&\'()*,.\\/:;?@[\\\]_{}'


def eval(fp, incl_punct=True):
    with open(fp, 'r') as f:
        no_correct_heads = 0
        no_correct_labels = 0
        no_correct_unlabeled_trees = 0
        no_correct_labeled_trees = 0
        total_token = 0
        total_trees = 0

        for line in f:
            data = json.loads(line.strip())

            total_trees += 1
            no_head_errors = 0
            no_label_errors = 0

            for k in range(len(data['words'])):
                if incl_punct or data['tags'][k] not in PUNCTS:
                    total_token += 1
                    if data['heads'][k] == data['pred_heads'][k]:
                        no_correct_heads += 1
                        if data['labels'][k] == data['pred_labels'][k]:
                            no_correct_labels += 1
                        else:
                            no_label_errors += 1
                    else:
                        no_head_errors += 1
                        no_label_errors += 1
            no_correct_unlabeled_trees += no_head_errors == 0
            no_correct_labeled_trees += no_label_errors == 0

        uas = 100. * no_correct_heads / total_token
        las = 100. * no_correct_labels / total_token
        uacc = 100. * no_correct_unlabeled_trees / total_trees
        lacc = 100. * no_correct_labeled_trees / total_trees

        print('UAS  = 100 * %d / %d = %.2f' % (no_correct_heads, total_token, uas))
        print('LAS  = 100 * %d / %d = %.2f' % (no_correct_labels, total_token, las))
        print('UACC = 100 * %d / %d = %.2f' % (no_correct_unlabeled_trees, total_trees, uacc))
        print('LACC = 100 * %d / %d = %.2f' % (no_correct_labeled_trees, total_trees, lacc))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parsing evaluation for JSONL format')
    parser.add_argument('input', type=str, help='Input file in JSONL format')
    parser.add_argument('-p', type=str2bool, default=True, help='Evaluate with punctuations')
    args = parser.parse_args()
    eval(args.input, args.p)
