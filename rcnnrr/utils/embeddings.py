import numpy as np


def read_embeddings(fp, mode='txt'):
    if mode == 'txt':
        return __read_embeddings_txt(fp)
    elif mode == 'bin':
        return __read_embeddings_bin(fp)
    else:
        raise ValueError('Do not understand \'%s\' mode' % mode)


def __read_embeddings_txt(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        vocab_size, dim = f.readline().rstrip().split()
        print('Read embeddings from file %s: vocab size is %s, dimension is %s' % (fp, vocab_size, dim))
        ids = list()
        embeddings = list()
        for i, line in enumerate(f):
            parts = line.rstrip().split(' ')
            ids.append(parts[0])
            embeddings.append(np.array(parts[1:], dtype='float32'))
            if i % 10000 == 0:
                print('Read row %d' % i, end='\r')
        print('Finish reading embeddings')
        embeddings = np.stack(embeddings, axis=0)
        print('Finish converting embeddings')
        return ids, embeddings


def __read_embeddings_bin(fp):
    raise NotImplementedError()


def write_embeddings(fp, words, embeddings, mode='txt'):
    if mode == 'txt':
        return __write_embeddings_txt(fp, words, embeddings)
    elif mode == 'bin':
        return __write_embeddings_bin(fp, words, embeddings)
    else:
        raise ValueError('Do not understand \'%s\' mode' % mode)


def __write_embeddings_txt(fp, words, embeddings):
    raise NotImplementedError()


def __write_embeddings_bin(fp, words, embeddings):
    raise NotImplementedError()
