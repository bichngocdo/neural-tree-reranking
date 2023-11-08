import argparse
import json
import sys


class BracketTokenizer:
    def __init__(self, s):
        self.string = s
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= len(self.string):
            raise StopIteration

        chars = list()
        while self.position < len(self.string):
            c = self.string[self.position]
            self.position += 1
            if c == '(' or c == ')':
                if len(chars) == 0:
                    return c
                else:
                    self.position -= 1
                    break
            if c == ' ' and len(chars) > 0:
                break
            if c != ' ':
                chars.append(c)

        return ''.join(chars)


class Stack(object):
    def __init__(self):
        self.items = list()

    def __len__(self):
        return len(self.items)

    def __str__(self):
        return self.items.__str__()

    def __iter__(self):
        return self.items.__iter__()

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]


def bracket2deptree(s):
    stack = Stack()
    head_stack = Stack()

    words = list()
    tags = list()
    head_map = dict()

    id = 1

    for token in BracketTokenizer(s):
        if token == '(':
            stack.push(token)
        elif token == ')':
            head = head_stack.pop()
            while stack.peek() != '(':
                item = stack.pop()
                head_map[item] = head
            if stack.peek() == '(':
                stack.pop()
            else:
                raise Exception('Expect \'(\'')

            stack.push(head)
        else:
            head_stack.push(id)
            id += 1
            if '/' in token:
                token = token.replace('\\/', '/')
                k = token.rfind('/')
                word = token[:k]
                tag = token[k + 1:]
                words.append(word)
                tags.append(tag)
            else:
                words.append(token)
                tags.append('_')

    for item in stack:
        head_map[item] = 0

    heads = list()
    for i in range(len(words)):
        heads.append(head_map[i + 1])

    return words, tags, heads


def convert(fp_gold, fp_sys, fp_out):
    with open(fp_gold, 'r') as f_gold, open(fp_sys, 'r') as f_sys, open(fp_out, 'w') as f_out:
        result = {
            'words': None,
            'tags': None,
            'heads': None,
            'labels': None,
            'nbest_parses': list(),
        }
        num_trees = -1
        count = 0

        while True:
            line_sys = f_sys.readline()
            if line_sys == '':
                break

            line_sys = line_sys.rstrip()
            if line_sys == '':
                continue
            elif line_sys.startswith('sent'):
                if num_trees > 0:
                    count += 1
                    if count % 100 == 0:
                        print(count)
                    assert len(result['nbest_parses']) == num_trees
                    json.dump(result, f_out)
                    f_out.write('\n')

                num_trees = int(line_sys.split()[1])
                line_gold = f_gold.readline()
                sent_gold = line_gold.rstrip()
                words, tags, heads = bracket2deptree(sent_gold)
                result['words'] = words
                result['tags'] = tags
                result['heads'] = heads
                result['labels'] = ['_'] * len(words)
                result['words'] = words
                del result['nbest_parses'][:]
            else:
                parts = line_sys.split('\t')
                score = float(parts[0])
                words, tags, heads = bracket2deptree(parts[1])
                parse = {
                    'heads': heads,
                    'labels': ['_'] * len(words),
                    'score': score
                }
                result['nbest_parses'].append(parse)

        if num_trees > 0:
            count += 1
            if count % 100 == 0:
                print(count)
            assert len(result['nbest_parses']) == num_trees
            json.dump(result, f_out)
            f_out.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert parsed k-best trees in bracket format to JSONL format')
    parser.add_argument('gold', type=str,
                        help='Input gold standard file in bracket format')
    parser.add_argument('input', type=str,
                        help='Input file containing k-best trees in bracket format')
    parser.add_argument('output', type=str,
                        help='Output file in JSONL format')
    args = parser.parse_args()

    convert(args.gold, args.input, args.output)
