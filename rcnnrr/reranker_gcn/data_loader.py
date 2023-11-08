import numpy as np

from rcnnrr.data.vocab import Vocab


def _recursive_convert_to_id(sequence, vocab):
    result = list()
    if not isinstance(sequence, list):
        return vocab[sequence]
    else:
        for item in sequence:
            result.append(_recursive_convert_to_id(item, vocab))
    return result


def _convert_to_ids(sentences, vocab, level=0):
    if level == 0:
        return vocab[sentences]
    elif level == 1:
        return [vocab[item] for item in sentences]
    elif level == 2:
        return [[vocab[item] for item in sentence] for sentence in sentences]
    elif level == 3:
        return [[[vocab[item] for item in items] for items in sentence] for sentence in sentences]
    else:
        raise Exception('List depth not supported')


class DataLoader(object):
    def __init__(self):
        self.word_vocab = None
        self.word_pt_vocab = None
        self.char_vocab = None
        self.tag_vocab = None
        self.label_vocab = None

        self.max_length = 0

        self.word_cutoff_threshold = 0
        self.lowercase = False

        self.NONE = '-NONE-'
        self.UNKNOWN = '-UNKNOWN-'
        self.ROOT = '-ROOT-'

    def init_pretrained_vocab(self, vocab_name, words):
        str2id = dict()
        id2str = list()
        id = 0

        for str in [self.NONE, self.UNKNOWN, self.ROOT]:
            str2id[str] = id
            id2str.append(str)
            id += 1

        for word in words:
            str2id[word] = id
            id2str.append(word)
            id += 1

        vocab = Vocab()
        vocab.str2id = str2id
        vocab.id2str = id2str
        vocab.unk_str = self.UNKNOWN
        vocab.unk_id = 1
        self.__setattr__(vocab_name, vocab)

    def load_embeddings(self, embeddings, have_unk=False):
        if have_unk:
            embeddings = np.pad(embeddings, ((2, 0), (0, 0)), 'constant', constant_values=0)
        else:
            embeddings = np.pad(embeddings, ((3, 0), (0, 0)), 'constant', constant_values=0)
        return embeddings

    def init_vocabs(self, raw_data):
        self.word_vocab = Vocab()
        self.char_vocab = Vocab()
        self.tag_vocab = Vocab()
        self.label_vocab = Vocab()

        words = raw_data['words']
        chars = raw_data['chars']
        tags = raw_data['tags']
        labels = raw_data['labels']

        if self.lowercase:
            old_words = words
            words = list()
            for item in old_words:
                if isinstance(item, list):
                    words.append([word.lower() for word in item])
                else:
                    words.append(item.lower())

        self.word_vocab.init(words, unk_str=self.UNKNOWN, special_strs=[self.NONE, self.ROOT],
                             cutoff_threshold=self.word_cutoff_threshold)
        self.char_vocab.init(chars, unk_str=self.UNKNOWN, special_strs=[self.NONE])
        self.tag_vocab.init(tags, unk_str=self.UNKNOWN, special_strs=[self.NONE, self.ROOT])
        self.label_vocab.init(labels, unk_str=self.UNKNOWN, special_strs=[self.NONE])

        self.max_length = max([len(s) for s in words])

    def load(self, raw_data):
        results = list()

        words = raw_data['words']
        chars = raw_data['chars']
        tags = raw_data['tags']

        if self.lowercase:
            old_words = words
            words = list()
            for item in old_words:
                if isinstance(item, list):
                    words.append([word.lower() for word in item])
                else:
                    words.append(item.lower())

        words_ = list()
        for sentence in words:
            words_.append([self.ROOT] + sentence)
        words = words_
        tags_ = list()
        for sentence in tags:
            tags_.append([self.ROOT] + sentence)
        tags = tags_
        chars_ = list()
        for sentence in chars:
            chars_.append([self.ROOT] + sentence)
        chars = chars_

        results.append(_convert_to_ids(words, self.word_vocab, level=2))
        if self.word_pt_vocab:
            results.append(_convert_to_ids(words, self.word_pt_vocab, level=2))
        else:
            results.append(None)

        results.append(_convert_to_ids(chars, self.char_vocab, level=3))
        results.append(_convert_to_ids(tags, self.tag_vocab, level=2))

        results.append(raw_data['heads'])
        results.append(_recursive_convert_to_id(raw_data['labels'], self.label_vocab))

        results.append(raw_data['nbest_tree_heads'])
        results.append(_recursive_convert_to_id(raw_data['nbest_tree_labels'], self.label_vocab))

        results.append(raw_data['nbest_in_edges'])
        results.append(_recursive_convert_to_id(raw_data['nbest_in_labels'], self.label_vocab))
        results.append(raw_data['nbest_out_edges'])
        results.append(_recursive_convert_to_id(raw_data['nbest_out_labels'], self.label_vocab))

        results.append(raw_data['nbest_scores'])
        results.append(raw_data['nbest_labels'])
        results.append(raw_data['nbest_margins'])

        return results
