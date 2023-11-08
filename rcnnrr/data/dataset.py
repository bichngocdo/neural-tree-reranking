from rcnnrr.data.batch_generators import SimpleBatch


class BatchIterator(object):
    def __init__(self, data, batch_indexes, data_converter):
        self.data = data
        self.batch_indexes = batch_indexes
        self.data_converter = data_converter
        self.position = -1

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.batch_indexes)

    def __next__(self):
        if self.position < len(self.batch_indexes) - 1:
            self.position += 1
            batch_idx = self.batch_indexes[self.position]
            batch = list()
            for element in self.data:
                if element is None:
                    batch.append(None)
                else:
                    batch.append([element[idx] for idx in batch_idx])
            if self.data_converter:
                return self.data_converter.convert(batch)
            else:
                return batch
        else:
            raise StopIteration


class Dataset(object):
    def __init__(self, data, raw_data=None, batch_generator=None, data_converter=None):
        self.data = data
        self.raw_data = raw_data
        if not batch_generator:
            size = len(data[0])
            batch_generator = SimpleBatch(size)
        self.batch_generator = batch_generator
        self.data_converter = data_converter

    def get_batches(self, batch_size, shuffle=False):
        batch_indexes = self.batch_generator.get_batch_indexes(batch_size, shuffle)
        return BatchIterator(self.data, batch_indexes, self.data_converter)
