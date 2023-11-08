import json


def write_conllx(fp, raw_data, results):
    with open(fp, 'w') as f:
        for i, result in enumerate(results):
            words = raw_data['words'][i]
            tags = raw_data['tags'][i]
            heads = raw_data['heads'][i]
            labels = raw_data['labels'][i]
            predicted_heads = raw_data['nbest_tree_heads'][i][result]
            predicted_labels = raw_data['nbest_tree_labels'][i][result]

            for j, (word, tag, head, label, predicted_head, predicted_label) in enumerate(
                    zip(words, tags, heads, labels, predicted_heads, predicted_labels)):
                f.write(f'{j + 1}\t{word}\t_\t_\t{tag}\t_\t{head}\t{label}\t{predicted_head}\t{predicted_label}\n')
            f.write('\n')


def write_json(fp, raw_data, predictions, scores):
    with open(fp, 'w') as f:
        for i, (prediction, all_score) in enumerate(zip(predictions, scores)):
            data = dict()
            data['words'] = raw_data['words'][i]
            data['tags'] = raw_data['tags'][i]
            data['heads'] = raw_data['heads'][i]
            data['labels'] = raw_data['labels'][i]

            data['pred_heads'] = raw_data['nbest_tree_heads'][i][prediction]
            data['pred_labels'] = raw_data['nbest_tree_labels'][i][prediction]

            data['nbest_parses'] = list()
            for heads, labels, old_score, new_score in zip(raw_data['nbest_tree_heads'][i],
                                                           raw_data['nbest_tree_labels'][i],
                                                           raw_data['nbest_scores'][i],
                                                           all_score):
                parse = dict()
                parse['heads'] = heads
                parse['labels'] = labels
                parse['score'] = old_score
                parse['rerank_score'] = float(new_score)
                data['nbest_parses'].append(parse)

            json.dump(data, f)
            f.write('\n')
