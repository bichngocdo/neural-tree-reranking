import argparse
import json


def json2conll(fp_in, fp_out):
    with open(fp_in, 'r') as f_in, open(fp_out, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.rstrip())
            items = ['_'] * 10
            for i in range(len(data['words'])):
                items[0] = str(i + 1)
                items[1] = data['words'][i]
                items[4] = data['tags'][i]
                items[6] = str(data['pred_heads'][i])
                items[7] = str(data['pred_labels'][i])
                f_out.write('\t'.join(items))
                f_out.write('\n')
            f_out.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert parsing results in JSONL to CoNLL format')
    parser.add_argument('input', type=str, help='Input file in JSONL format')
    parser.add_argument('output', type=str, help='Output file in CoNLL format')
    args = parser.parse_args()

    json2conll(args.input, args.output)
