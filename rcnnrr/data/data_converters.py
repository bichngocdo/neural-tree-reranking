import numpy as np


def _nested_compute_shape(sequence, i, max_lengths):
    if not isinstance(sequence, list):
        if i in max_lengths and max_lengths[i] is not None:
            raise Exception('Invalid sequence:', sequence)
        max_lengths[i] = None
        return
    for s in sequence:
        _nested_compute_shape(s, i + 1, max_lengths)
    max_length = max_lengths.setdefault(i, 0)
    if max_length is None:
        raise Exception('Invalid sequence:', sequence)
    if len(sequence) > max_length:
        max_lengths[i] = len(sequence)


def _compute_shape(sequence):
    max_lengths = dict()
    _nested_compute_shape(sequence, 0, max_lengths)
    shape = list()
    for dim in range(len(max_lengths)):
        max_length = max_lengths[dim]
        if max_length is not None:
            shape.append(max_length)
    return shape


def _nested_copy2tensor(tensor, sequence, indexes):
    if len(sequence) > 0 and not isinstance(sequence[0], list):
        tensor[tuple(indexes + [slice(None, len(sequence), None)])] = sequence
        return
    for i, s in enumerate(sequence):
        indexes.append(i)
        _nested_copy2tensor(tensor, s, indexes)
        del indexes[-1]


def _copy2tensor(tensor, sequence):
    indexes = list()
    _nested_copy2tensor(tensor, sequence, indexes)


def convert_to_tensor(sequences, type='int32', value=0):
    shape = _compute_shape(sequences)
    tensor = np.zeros(shape).astype(type) + value
    _copy2tensor(tensor, sequences)
    return tensor


class AbstractDataConverter(object):
    def __init__(self):
        pass

    def convert(self, batch):
        raise NotImplementedError
