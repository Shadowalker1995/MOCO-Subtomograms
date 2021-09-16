"""
FileName:	sp.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-09-15 23:31:00
"""


import numpy as np
import os
import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading

# how many labeled points should we have in training?
# numlabels = 20
numlabels = 5

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
def label_propagation(outputs, targets, stage):
    outputs = np.load(f'Figures/{stage}_outputs.npy')
    targets = np.load(f'Figures/{stage}_targets.npy')
    if os.path.exists(f'./Figures/{stage}_outputs_tsne.npy'):
        outputs_tsne = np.load(f'./Figures/{stage}_outputs_tsne.npy')
    else:
        outputs_pca50 = PCA(n_components=50).fit_transform(outputs)
        outputs_tsne = TSNE(random_state=RS).fit_transform(outputs_pca50)
        np.save(f'./Figures/{stage}_outputs_tsne.npy', outputs_tsne)

    # outputs_tsne = outputs

    labels = np.copy(targets)

    if stage == 'train':
        idx = [rng.choice(1800, 1800 - numlabels, replace=False) + i * 1800 for i in range(5)]
    elif stage == 'val':
        idx = [rng.choice(200, 200 - numlabels, replace=False) + i * 200 for i in range(5)]
    random_unlabeled_points = np.concatenate(idx, axis=0)
    labels[random_unlabeled_points] = -1

    label_prop_model.fit(outputs_tsne, labels)
    pred = label_prop_model.predict(outputs_tsne)
    print(label_prop_model.score(outputs_tsne, targets))
    return pred


if __name__ == "__main__":
    outputs = np.load('Figures/train_outputs.npy')
    targets = np.load('Figures/train_targets.npy')
    pred = label_propagation(outputs, targets, 'val')
