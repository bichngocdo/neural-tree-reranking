import argparse

import numpy as np
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='Effective Dimensionality Reduction for Word Embeddings')
parser.add_argument('input', type=str, help='input embeddings')
parser.add_argument('output', type=str, help='output embeddings')
parser.add_argument('-n', '--size', type=int, help='size of the output embeddings')

args = parser.parse_args()
print(args)

Glove = {}
f = open(args.input)

print('Loading input vectors.')
line = f.readline().rstrip()
parts = line.split(' ')
num_words = int(parts[0])
input_dim = int(parts[1])

for line in f:
    values = line.rstrip().split(' ')
    word = values[0]
    coefs = np.array(values[1:], dtype='float32')
    assert input_dim == len(coefs)
    Glove[word] = coefs
f.close()
print('Done.')

X_train = []
X_train_names = []
for x in Glove:
    X_train.append(Glove[x])
    X_train_names.append(x)

X_train = np.asarray(X_train)
pca_embeddings = {}

# PCA to Get Top Components
pca = PCA(n_components=input_dim)
X_train = X_train - np.mean(X_train)
X_fit = pca.fit_transform(X_train)
U1 = pca.components_

z = []

# Removing Projections on Top Components
for i, x in enumerate(X_train):
    for u in U1[0:7]:
        x = x - np.dot(u.transpose(), x) * u
    z.append(x)

z = np.asarray(z)

output_dim = args.size if args.size is not None else input_dim // 2
# PCA Dim Reduction
pca = PCA(n_components=output_dim)
X_train = z - np.mean(z)
X_new_final = pca.fit_transform(X_train)

# PCA to do Post-Processing Again
pca = PCA(n_components=output_dim)
X_new = X_new_final - np.mean(X_new_final)
X_new = pca.fit_transform(X_new)
Ufit = pca.components_

X_new_final = X_new_final - np.mean(X_new_final)

final_pca_embeddings = {}
with open(args.output, 'w') as embedding_file:
    embedding_file.write('%d %d\n' % (num_words, output_dim))
    for i, x in enumerate(X_train_names):
        final_pca_embeddings[x] = X_new_final[i]
        embedding_file.write('%s ' % x)
        for u in Ufit[0:7]:
            final_pca_embeddings[x] = final_pca_embeddings[x] - np.dot(u.transpose(), final_pca_embeddings[x]) * u

        coefs = ['%.9f' % t for t in final_pca_embeddings[x]]
        embedding_file.write(' '.join(coefs))
        embedding_file.write('\n')
