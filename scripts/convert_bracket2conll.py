import argparse


class BracketTokenizer(object):
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


def bracket2conll(fp_in, fp_out):
    with open(fp_in, 'r') as f_in, open(fp_out, 'w') as f_out:
        for line in f_in:
            words, tags, heads = bracket2deptree(line.rstrip())
            for i, (word, tag, head) in enumerate(zip(words, tags, heads)):
                data = ['_'] * 10
                data[0] = str(i + 1)
                data[1] = word
                data[4] = tag
                data[6] = str(head)
                f_out.write('\t'.join(data))
                f_out.write('\n')
            f_out.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert parsing results in bracket format to CoNLL format')
    parser.add_argument('input', type=str, help='Input file in bracket format')
    parser.add_argument('output', type=str, help='Output file in CoNLL format')
    args = parser.parse_args()

    bracket2conll(args.input, args.output)
