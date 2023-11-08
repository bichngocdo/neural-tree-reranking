def flatten_list(l):
    if isinstance(l, list):
        for seq in l:
            for item in flatten_list(seq):
                yield item
    else:
        yield l


class Vocab(object):
    def __init__(self):
        self.str2id = None
        self.id2str = None
        self.unk_str = None
        self.unk_id = None
        self.freqs = None

    def init(self, sentences, unk_str='<UNK>', special_strs=None, cutoff_threshold=0):
        special_strs = special_strs if special_strs else list()
        self.unk_str = unk_str
        self.unk_id = len(special_strs)
        special_strs.append(unk_str)

        self.freqs = dict()
        for item in flatten_list(sentences):
            self.freqs.setdefault(item, 0)
            self.freqs[item] += 1

        self.str2id = dict()
        self.id2str = list()
        id = 0
        for item in special_strs:
            self.str2id[item] = id
            self.id2str.append(item)
            id += 1
        for item, count in self.freqs.items():
            if count >= cutoff_threshold:
                self.str2id[item] = id
                self.id2str.append(item)
                id += 1

    def extend(self, sentences, cutoff_threshold=0):
        for item in flatten_list(sentences):
            self.freqs.setdefault(item, 0)
            self.freqs[item] += 1
        id = len(self.str2id)
        for item, count in self.freqs.iteritems():
            if count >= cutoff_threshold and item not in self.str2id:
                self.str2id[item] = id
                self.id2str.append(item)
                id += 1

    def __getitem__(self, item):
        return self.str2id.get(item, self.unk_id)

    def __len__(self):
        return len(self.str2id)

    def lookup(self, id):
        return self.id2str[id]
