import json
import random

from collections import deque


def _bfs(heads, root=0):
    n = len(heads)
    visited = set()
    result = list()
    time = [-1] * (len(heads) + 1)
    queue = deque()
    queue.append(root)
    t = 0
    while len(visited) <= n:
        while len(queue) > 0:
            current = queue.popleft()
            visited.add(current)
            result.append(current)
            time[current] = t
            t += 1
            for i in range(n):
                if heads[i] == current and (i + 1) not in visited:
                    queue.append(i + 1)
        for i in range(n):
            if (i + 1) not in visited:
                queue.append(i + 1)
                break
    return result, time


def _extract_tree_info(heads):
    head_child_map = dict()
    for i, head in enumerate(heads):
        child = i + 1
        head_child_map.setdefault(head, list()).append(child)
    bfs_order, _ = _bfs(heads, 0)
    result_heads = list()
    result_children = list()
    for node in bfs_order[::-1]:
        if node in head_child_map:
            result_heads.append(node)
            result_children.append(head_child_map[node])
    return result_heads, result_children


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
            'nbest_trees': list(),
            'nbest_tree_labels': list(),
            'nbest_heads': list(),
            'nbest_children': list(),
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
            has_correct_tree = False
            for i, tree in enumerate(data['nbest_parses']):
                uas, las = _score(data['heads'], data['labels'], tree['heads'], tree['labels'])
                scores.append((uas, las))
                if uas == 1 and las == 1:
                    has_gold_tree = True
                if uas == 1:
                    has_correct_tree = True
                    labels.append(1)
                else:
                    labels.append(0)

            if not (has_correct_tree or keep_all or add_oracle):
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

            nb_trees = list()
            nb_tree_labels = list()
            nb_heads = list()
            nb_children = list()
            nb_scores = list()
            nb_labels = list()
            nb_margins = list()

            for i, tree in enumerate(data['nbest_parses']):
                nb_trees.append(tree['heads'])
                nb_tree_labels.append(tree['labels'])
                head, children = _extract_tree_info(tree['heads'])
                nb_heads.append(head)
                nb_children.append(children)
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

            max_correct_heads += current_max_correct_heads
            min_correct_heads += current_min_correct_heads
            max_correct_labels += current_max_correct_labels
            min_correct_labels += current_min_correct_labels

            ids = list(range(len(nb_trees)))

            random.shuffle(ids)
            nb_trees = [nb_trees[i] for i in ids]
            nb_tree_labels = [nb_tree_labels[i] for i in ids]
            nb_heads = [nb_heads[i] for i in ids]
            nb_children = [nb_children[i] for i in ids]
            nb_scores = [nb_scores[i] for i in ids]
            nb_labels = [nb_labels[i] for i in ids]
            nb_margins = [nb_margins[i] for i in ids]
            results['nbest_trees'].append(nb_trees)
            results['nbest_tree_labels'].append(nb_tree_labels)
            results['nbest_heads'].append(nb_heads)
            results['nbest_children'].append(nb_children)
            results['nbest_scores'].append(nb_scores)
            results['nbest_labels'].append(nb_labels)
            results['nbest_margins'].append(nb_margins)

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
        print(f'Best UAS             : {max_correct_heads} / {no_tokens}'
              f' = {best_uas:5.2f}')
        print(f'Worst UAS            : {min_correct_heads} / {no_tokens}'
              f' = {worst_uas:5.2f}')
        print(f'Best LAS             : {max_correct_labels} / {no_tokens}'
              f' = {best_las:5.2f}')
        print(f'Worst LAS            : {min_correct_labels} / {no_tokens}'
              f' = {worst_las:5.2f}')

        return results
