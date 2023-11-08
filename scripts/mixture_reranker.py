import argparse
import json

import numpy as np


def raw_metric(gold_heads, gold_labels, heads, labels):
    uas = 0
    las = 0
    for gh, gl, h, l in zip(gold_heads, gold_labels, heads, labels):
        if gh == h:
            uas += 1
            if gl == l:
                las += 1
    return uas, las


def find_alpha(fp, alpha_range=(0, 1), step=0.1, small=1e-12):
    alpha_min, alpha_max = alpha_range

    results = dict()
    for alpha in np.arange(alpha_min, alpha_max + 1e-7, step):
        results[alpha] = dict()
        results[alpha]['uas'] = 0
        results[alpha]['las'] = 0
    total_tokens = 0

    with open(fp, 'r') as f:
        for line in f:
            mixture_scores = dict()

            data = json.loads(line.rstrip())
            total_tokens += len(data['words'])

            for parse in data['nbest_parses']:
                for alpha in np.arange(alpha_min, alpha_max + 1e-7, step):
                    # if parse['score'] <= 0:
                    #     parse['score'] = small
                    # score = alpha * parse['rerank_score'] + (1 - alpha) * np.log(parse['score'])
                    score = alpha * parse['rerank_score'] + (1 - alpha) * parse['score']
                    mixture_scores.setdefault(alpha, list()).append(score)

            for alpha in np.arange(alpha_min, alpha_max + 1e-7, step):
                best_score = float('-inf')
                best_tree = -1

                for i, score in enumerate(mixture_scores[alpha]):
                    if score > best_score:
                        best_score = score
                        best_tree = i

                uas, las = raw_metric(data['heads'], data['labels'],
                                      data['nbest_parses'][best_tree]['heads'],
                                      data['nbest_parses'][best_tree]['labels'])

                results[alpha]['uas'] += uas
                results[alpha]['las'] += las

    print('alpha  | UAS         | LAS         ')
    print('------+-------------+-------------')
    for alpha in np.arange(alpha_min, alpha_max + 1e-7, step):
        uas = 100. * results[alpha]['uas'] / total_tokens
        las = 100. * results[alpha]['las'] / total_tokens
        print('%5.4f | %11.2f | %11.2f ' % (alpha, uas, las))


def combine(fp_in, fp_out, alpha):
    with open(fp_in, 'r') as f_in, open(fp_out, 'w') as f_out:
        raw_uas = 0
        raw_las = 0
        total_tokens = 0

        for line in f_in:
            data = json.loads(line.rstrip())
            total_tokens += len(data['words'])

            mixture_scores = list()
            for parse in data['nbest_parses']:
                parser_score = parse['score']
                score = alpha * parse['rerank_score'] + (1 - alpha) * parser_score
                parse['mixture_score'] = score
                mixture_scores.append(score)

            best_score = float('-inf')
            best_tree = -1

            for i, score in enumerate(mixture_scores):
                if score > best_score:
                    best_score = score
                    best_tree = i

            data['pred_heads'] = data['nbest_parses'][best_tree]['heads']
            data['pred_labels'] = data['nbest_parses'][best_tree]['labels']

            uas, las = raw_metric(data['heads'], data['labels'],
                                  data['pred_heads'], data['pred_labels'])
            raw_uas += uas
            raw_las += las

            json.dump(data, f_out)
            f_out.write('\n')

        uas = 100. * raw_uas / total_tokens
        las = 100. * raw_las / total_tokens
        print('UAS = %5.2f' % uas)
        print('LAS = %5.2f' % las)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mixture reranker')
    subparsers = parser.add_subparsers()

    parser_find = subparsers.add_parser('find',
                                        description='Find the mixture parameter')
    parser_find.set_defaults(command='find')
    parser_find.add_argument('input', type=str,
                             help='Input file in JSONL format')
    parser_find.add_argument('-r', '--range', type=float, nargs=2, default=(0, 1),
                             help='alpha range')
    parser_find.add_argument('-s', '--step', type=float, default=0.1,
                             help='alpha step')

    parser_combine = subparsers.add_parser('combine',
                                           description='Combine ranking results with a mixture parameter')
    parser_combine.set_defaults(command='combine')
    parser_combine.add_argument('input', type=str,
                                help='Input file in JSONL format')
    parser_combine.add_argument('output', type=str,
                                help='Output file in JSONL format')
    parser_combine.add_argument('-a', '--alpha', type=float, required=True,
                                help='alpha')

    args = parser.parse_args()

    if args.command == 'find':
        find_alpha(args.input, args.range, args.step)
    elif args.command == 'combine':
        combine(args.input, args.output, args.alpha)
