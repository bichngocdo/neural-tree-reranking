import argparse
import json


class BlockFile:
    def __init__(self, f):
        self.file = f
        self.sentence = list()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            line = self.file.readline()
            if not line:
                if not self.sentence:
                    raise StopIteration
                else:
                    sentences = self.sentence
                    self.sentence = []
                    return sentences
            else:
                line = line.rstrip()
                if not line and self.sentence:
                    sentences = self.sentence
                    self.sentence = []
                    return sentences
                if line:
                    self.sentence.append(line)

    def close(self):
        self.file.close()


def convert(fp_in, fp_out):
    with open(fp_in, 'r') as f_in, open(fp_out, 'w') as f_out:
        f_in = BlockFile(f_in)
        for id, block in enumerate(f_in):
            nbest_parse_line = block[-1]
            if not nbest_parse_line.startswith('ParseNBest'):
                raise Exception('No ParseNBest: ' + nbest_parse_line)
            _, n, parses = nbest_parse_line.split('\t')
            parses = parses.split(';')
            assert len(parses) == int(n)
            trees = list()
            scores = list()
            for parse in parses:
                score = 0.
                parts = parse[1:-1].split(' @@@ ')

                for part in parts[0].split(' '):
                    key, value = part.split('=')
                    if 'Score' in key:
                        score = float(value)
                        break
                parse_parts = parts[1].split(' ')

                tree = list()
                for part in parse_parts[1:]:
                    head, label = part.split(',')
                    tree.append((int(head), label))
                trees.append(tree)
                scores.append(score)

            result = {
                'words': list(),
                'tags': list(),
                'heads': list(),
                'labels': list(),
                'nbest_parses': list(),
            }
            for line in block[:-1]:
                parse_parts = line.split('\t')
                result['words'].append(parse_parts[1])
                result['tags'].append(parse_parts[5])
                result['heads'].append(int(parse_parts[8]))
                result['labels'].append(parse_parts[10])
            for tree, score in zip(trees, scores):
                parse = {
                    'heads': list(),
                    'labels': list(),
                    'score': 0.
                }
                for head, label in tree:
                    parse['heads'].append(head)
                    parse['labels'].append(label)
                parse['score'] = score
                result['nbest_parses'].append(parse)

            json.dump(result, f_out)
            f_out.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert parsed k-best trees to JSONL format')
    parser.add_argument('input', type=str,
                        help='Input file containing k-best trees output by the MATE parser')
    parser.add_argument('output', type=str,
                        help='Output file in JSONL format')
    args = parser.parse_args()

    convert(args.input, args.output)
