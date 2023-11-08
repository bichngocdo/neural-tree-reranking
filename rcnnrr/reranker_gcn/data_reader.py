import json
import random


def _extract_info(heads, labels):
    head_child_map = dict()
    for i, head in enumerate(heads):
        child = i + 1
        head_child_map.setdefault(head, list()).append(child)
    in_edges = [[-1]] + [[h] for h in heads]
    in_labels = [[]] + [[l] for l in labels]
    out_edges = list()
    out_labels = list()
    for node in range(len(heads) + 1):
        if node in head_child_map:
            out_edges.append(head_child_map[node])
            out_labels.append([labels[i - 1] for i in head_child_map[node]])
        else:
            out_edges.append([])
            out_labels.append([])

    return in_edges, in_labels, out_edges, out_labels


def _compute_margin(gold_heads, heads):
    margin = 0
    for gold_head, head in zip(gold_heads, heads):
        if head != gold_head:
            margin += 1
    return margin


def _score(gold_heads, gold_labels, heads, labels):
    uas = 0
    las = 0
    total = 0
    for gh, gl, h, l in zip(gold_heads, gold_labels, heads, labels):
        total += 1
        if gh == h:
            uas += 1
            if gl == l:
                las += 1
    uas /= total
    las /= total

    return uas, las


def read(fp, add_oracle=False, keep_all=False, is_training=False, use_structured_margin=False):
    with open(fp, 'r') as f:
        results = {
            'words': list(),
            'chars': list(),
            'tags': list(),
            'heads': list(),
            'labels': list(),
            'nbest_tree_heads': list(),
            'nbest_tree_labels': list(),
            'nbest_in_edges': list(),
            'nbest_in_labels': list(),
            'nbest_out_edges': list(),
            'nbest_out_labels': list(),
            'nbest_scores': list(),
            'nbest_labels': list(),
            'nbest_margins': list()
        }

        no_removed_instances = 0
        no_kept_instances = 0
        no_correct = 0

        no_tokens = 0
        min_correct_heads = 0
        max_correct_heads = 0
        min_correct_labels = 0
        max_correct_labels = 0

        for line in f:
            data = json.loads(line.strip())

            scores = list()
            labels = list()
            has_gold_tree = False
            for i, tree in enumerate(data['nbest_parses']):
                uas, las = _score(data['heads'], data['labels'], tree['heads'], tree['labels'])
                scores.append((uas, las))
                if uas == 1 and las == 1:
                    has_gold_tree = True
                    labels.append(1)
                else:
                    labels.append(0)

            if not (has_gold_tree or keep_all or add_oracle):
                no_removed_instances += 1
                continue

            no_kept_instances += 1
            if add_oracle and not has_gold_tree:
                data['nbest_parses'].append({
                    'heads': data['heads'],
                    'labels': data['labels'],
                    'score': -1
                })
                scores.append((1., 1.))
                labels.append(1)

            if is_training:
                correct_trees = [i for i in range(len(labels)) if labels[i] == 1]
                if len(correct_trees) > 1:
                    best_score = max([scores[i] for i in correct_trees])
                    removed_trees = [i for i in correct_trees if scores[i] < best_score]
                    data['nbest_parses'] = [tree for i, tree in enumerate(data['nbest_parses'])
                                            if i not in removed_trees]
                    labels = [label for i, label in enumerate(labels)
                              if i not in removed_trees]

            no_correct += has_gold_tree

            no_tokens += len(data['heads'])

            current_max_correct_heads = 0
            current_min_correct_heads = float('inf')
            current_max_correct_labels = 0
            current_min_correct_labels = float('inf')

            results['words'].append(data['words'])
            results['chars'].append([[c for c in word] for word in data['words']])
            results['tags'].append(data['tags'])
            results['heads'].append(data['heads'])
            results['labels'].append(data['labels'])

            nb_tree_heads = list()
            nb_tree_labels = list()
            nb_in_edges = list()
            nb_in_labels = list()
            nb_out_edges = list()
            nb_out_labels = list()
            nb_scores = list()
            nb_labels = list()
            nb_margins = list()

            for i, tree in enumerate(data['nbest_parses']):
                nb_tree_heads.append(tree['heads'])
                nb_tree_labels.append(tree['labels'])
                in_edges, in_labels, out_edges, out_labels = _extract_info(tree['heads'], tree['labels'])
                nb_in_edges.append(in_edges)
                nb_in_labels.append(in_labels)
                nb_out_edges.append(out_edges)
                nb_out_labels.append(out_labels)
                nb_scores.append(tree['score'])
                nb_labels.append(labels[i])
                margin = _compute_margin(data['heads'], tree['heads'])
                if use_structured_margin or margin == 0:
                    nb_margins.append(margin)
                else:
                    nb_margins.append(1)
                num_correct_heads = 0
                num_correct_labels = 0
                for h, p_h, l, p_l in zip(data['heads'], tree['heads'], data['labels'], tree['labels']):
                    if h == p_h:
                        num_correct_heads += 1
                        if l == p_l:
                            num_correct_labels += 1

                if num_correct_heads > current_max_correct_heads:
                    current_max_correct_heads = num_correct_heads
                if num_correct_heads < current_min_correct_heads:
                    current_min_correct_heads = num_correct_heads
                if num_correct_labels > current_max_correct_labels:
                    current_max_correct_labels = num_correct_labels
                if num_correct_labels < current_min_correct_labels:
                    current_min_correct_labels = num_correct_labels

            ids = list(range(len(nb_tree_heads)))

            random.shuffle(ids)
            nb_tree_heads = [nb_tree_heads[i] for i in ids]
            nb_tree_labels = [nb_tree_labels[i] for i in ids]
            nb_in_edges = [nb_in_edges[i] for i in ids]
            nb_in_labels = [nb_in_labels[i] for i in ids]
            nb_out_edges = [nb_out_edges[i] for i in ids]
            nb_out_labels = [nb_out_labels[i] for i in ids]
            nb_scores = [nb_scores[i] for i in ids]
            nb_labels = [nb_labels[i] for i in ids]
            nb_margins = [nb_margins[i] for i in ids]
            results['nbest_tree_heads'].append(nb_tree_heads)
            results['nbest_tree_labels'].append(nb_tree_labels)
            results['nbest_in_edges'].append(nb_in_edges)
            results['nbest_in_labels'].append(nb_in_labels)
            results['nbest_out_edges'].append(nb_out_edges)
            results['nbest_out_labels'].append(nb_out_labels)
            results['nbest_scores'].append(nb_scores)
            results['nbest_labels'].append(nb_labels)
            results['nbest_margins'].append(nb_margins)

            max_correct_heads += current_max_correct_heads
            min_correct_heads += current_min_correct_heads
            max_correct_labels += current_max_correct_labels
            min_correct_labels += current_min_correct_labels

        original_acc = 100. * no_correct / (no_kept_instances + no_removed_instances)
        filtered_acc = 100. * no_correct / no_kept_instances
        best_uas = 100. * max_correct_heads / no_tokens
        worst_uas = 100. * min_correct_heads / no_tokens
        best_las = 100. * max_correct_labels / no_tokens
        worst_las = 100. * min_correct_labels / no_tokens

        print(f'No. removed instances: {no_removed_instances}')
        print(f'Original accuracy    : {no_correct} / {no_kept_instances + no_removed_instances}'
              f' = {original_acc:5.2f}')
        print(f'Filtered accuracy    : {no_correct} / {no_kept_instances}'
              f' = {filtered_acc:5.2f}')
        print(f'Best UAS : {max_correct_heads} / {no_tokens} = {best_uas:5.2f}')
        print(f'Worst UAS: {min_correct_heads} / {no_tokens} = {worst_uas:5.2f}')
        print(f'Best LAS : {max_correct_labels} / {no_tokens} = {best_las:5.2f}')
        print(f'Worst LAS: {min_correct_labels} / {no_tokens} = {worst_las:5.2f}')

        return results
