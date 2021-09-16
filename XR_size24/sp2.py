import numpy as np
import pickle
import os
import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading

# def pickley(filename):
#     with open(filename, 'rb') as pickle_file:
#         return pickle.load(pickle_file, encoding='latin1')

# d = pickley('PICA_py3/fts.pickle')
# x = d['x']
# y = d['y']

# how many labeled points should we have in training?
# numlabels = 20
numlabels = 20

RS = 42
rng = np.random.RandomState(RS)


def rbf_kernel_safe(X, Y=None, gamma=None):
    X, Y = sklearn.metrics.pairwise.check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = sklearn.metrics.pairwise.euclidean_distances(X, Y, squared=True)
    K *= -gamma
    K -= K.max()
    np.exp(K, K)  # exponentiate K in-place
    return K


# define label propagation and train
# label_prop_model = LabelPropagation(kernel='rbf', tol=0.01, gamma=20)
label_prop_model = LabelSpreading(kernel='rbf')


# Label Propagation
def label_propagation():
    train_outputs = np.load('Figures/train_outputs.npy')
    train_targets = np.load('Figures/train_targets.npy')
    val_outputs = np.load('Figures/val_outputs.npy')
    val_targets = np.load('Figures/val_targets.npy')
    if os.path.exists('./Figures/train_outputs_tsne.npy'):
        train_outputs_tsne = np.load('./Figures/train_outputs_tsne.npy')
    else:
        train_outputs_pca50 = PCA(n_components=50).fit_transform(train_outputs)
        train_outputs_tsne = TSNE(random_state=RS).fit_transform(train_outputs_pca50)
        np.save('./Figures/train_outputs_tsne.npy', train_outputs_tsne)
    if os.path.exists('./Figures/val_outputs_tsne.npy'):
        val_outputs_tsne = np.load('./Figures/val_outputs_tsne.npy')
    else:
        val_outputs_pca50 = PCA(n_components=50).fit_transform(val_outputs)
        val_outputs_tsne = TSNE(random_state=RS).fit_transform(val_outputs_pca50)
        np.save('./Figures/val_outputs_tsne.npy', val_outputs_tsne)

    labels = np.copy(train_targets)

    idx = [rng.choice(1800, 1800 - numlabels, replace=False) + i * 1800 for i in range(5)]
    random_unlabeled_points = np.concatenate(idx, axis=0)
    labels[random_unlabeled_points] = -1

    label_prop_model.fit(train_outputs_tsne, labels)
    pred = label_prop_model.predict(val_outputs_tsne)
    print(label_prop_model.score(train_outputs_tsne, train_targets))
    return pred


if __name__ == "__main__":
    pred = label_propagation()
