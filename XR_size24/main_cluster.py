"""
FileName:	main_cluster.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-08-04 18:52:30
"""


import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import math

import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import Encoder3D.Model_RB3D
import Encoder3D.Model_DSRF3D_v2
import Encoder3D.Model_YOPO
import Custom_CryoET_DataLoader
from CustomTransforms import ToTensor, Normalize3D

model_names = ['RB3D', 'DSRF3D_v2', 'YOPO']

Encoders3D_dictionary = {'RB3D': Encoder3D.Model_RB3D.RB3D, 'DSRF3D_v2': Encoder3D.Model_DSRF3D_v2.DSRF3D_v2,
                         'YOPO': Encoder3D.Model_YOPO.YOPO}

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='RB3D',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: RB3D)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='visualize on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')

parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    def model_init():
        args.gpu = gpu
        # suppress printing if not master
        if args.multiprocessing_distributed and args.gpu != 0:
            def print_pass(*args):
                pass

            builtins.print = print_pass

        if args.gpu is not None:
            print("Use GPU: {} for training".format(args.gpu))

        if args.distributed:
            if args.dist_url == "env://" and args.rank == -1:
                args.rank = int(os.environ["RANK"])
            if args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                args.rank = args.rank * ngpus_per_node + gpu
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
        # create model
        print("=> creating model '{}'".format(args.arch))
        model = Encoders3D_dictionary[args.arch](num_classes=args.moco_dim, keepfc=True)

        # freeze all layers
        for name, param in model.named_parameters():
            param.requires_grad = False

        # load from pre-trained, before DistributedDataParallel constructor
        if args.pretrained:
            if os.path.isfile(args.pretrained):
                print("=> loading checkpoint '{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained, map_location="cpu")

                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q'):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed k
                    del state_dict[k]

                args.start_epoch = 0
                msg = model.load_state_dict(state_dict, strict=False)
                print(msg)

                print("=> loaded pre-trained model '{}'".format(args.pretrained))
            else:
                print("=> no checkpoint found at '{}'".format(args.pretrained))

        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

        cudnn.benchmark = True
        return model

    # args.evaluate = False
    filename = '10_2000_30_01.pickle'
    sub_dir = os.path.basename(args.pretrained)[:-8]
    mkdir(f'./Figures/{sub_dir}/')
    num_labels = 20

    if args.evaluate:
        stage = 'val'
        stage_normalize = Normalize3D(mean=[-0.00086651], std=[0.90188922])
    else:
        stage = 'train'
        stage_normalize = Normalize3D(mean=[-0.00076087], std=[0.90214654])
    stage_transforms = transforms.Compose([
        ToTensor(),
        # stage_normalize,
    ])

    if os.path.exists(f'./Figures/{sub_dir}/{stage}_outputs.npy'):
        outputs = np.load(f'./Figures/{sub_dir}/{stage}_outputs.npy')
        targets = np.load(f'./Figures/{sub_dir}/{stage}_targets.npy')
        print(f"=> the shape of extracted features for {stage} set is: ", outputs.shape)
        print(f"=> the shape of labels for {stage} set is: ", targets.shape)
    else:
        model = model_init()
        # Data loading code
        dataset = Custom_CryoET_DataLoader.CryoETDatasetLoader(
            filename, stage=stage,
            transform=stage_transforms)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        outputs, targets = validate(loader, model, args)
        np.save(f'./Figures/{sub_dir}/{stage}_outputs.npy', outputs)
        np.save(f'./Figures/{sub_dir}/{stage}_targets.npy', targets)
    visualize(outputs, targets, stage, sub_dir)
    clustering(outputs, targets, stage, sub_dir)
    label_spreading(outputs, targets, num_labels, stage, sub_dir)
    label_propagation(outputs, targets, num_labels, stage, sub_dir)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        # print("---  new folder...  ---")


def validate(val_loader, model, args):
    # switch to evaluate mode
    model.eval()

    outputs = []
    targets = []

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            outputs.append(output)
            targets.append(target)

    # (4500, 128)
    outputs = torch.cat(outputs, dim=0).cpu().numpy()
    # (4500,)
    targets = torch.cat(targets, dim=0).cpu().numpy()

    return outputs, targets


def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


# Visualization
def visualize(outputs, targets, stage, sub_dir):
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    # print(len(outputs))
    # print(outputs[0].shape)
    RS = 123

    # pca + t-sne
    if os.path.exists(f'./Figures/{sub_dir}/{stage}_outputs_tsne.npy'):
        outputs_tsne = np.load(f'./Figures/{sub_dir}/{stage}_outputs_tsne.npy')
    else:
        outputs_pca50 = PCA(n_components=50).fit_transform(outputs)
        outputs_tsne = TSNE(random_state=RS).fit_transform(outputs_pca50)
        np.save(f'./Figures/{sub_dir}/{stage}_outputs_tsne.npy', outputs_tsne)

    plt.figure()
    fashion_scatter(outputs_tsne, targets)
    plt.savefig(f'./Figures/{sub_dir}/{stage}_fashion_scatter_pca+tsne.png')
    plt.figure()
    sns.scatterplot(x=outputs_tsne[:, 0], y=outputs_tsne[:, 1], hue=targets, legend='full', palette='hls')
    plt.savefig(f'./Figures/{sub_dir}/{stage}_scatter_pca+tsne.png')

    # only pca
    if os.path.exists(f'./Figures/{sub_dir}/{stage}_outputs_pca4.npy'):
        outputs_pca4 = np.load(f'./Figures/{sub_dir}/{stage}_outputs_pca4.npy')
    else:
        pca4 = PCA(n_components=4)
        outputs_pca4 = pca4.fit_transform(outputs)
        np.save(f'./Figures/{sub_dir}/{stage}_outputs_pca4.npy', outputs_pca4)
        print('Variance explained per principal component: {}'.format(pca4.explained_variance_ratio_))
    # taking first and second principal component
    outputs_pca2 = outputs_pca4[:, :2]

    plt.figure()
    fashion_scatter(outputs_pca2, targets)
    plt.savefig(f'./Figures/{sub_dir}/{stage}_fashion_scatter_pca.png')
    plt.figure()
    sns.scatterplot(x=outputs_pca2[:, 0], y=outputs_pca2[:, 1], hue=targets, legend='full', palette='hls')
    plt.savefig(f'./Figures/{sub_dir}/{stage}_scatter_pca.png')


# Clustering
def clustering(outputs, targets, stage, sub_dir):
    RS = 123
    if os.path.exists(f'./Figures/{sub_dir}/{stage}_outputs_tsne.npy'):
        outputs_tsne = np.load(f'./Figures/{sub_dir}/{stage}_outputs_tsne.npy')
    else:
        outputs_pca50 = PCA(n_components=50).fit_transform(outputs)
        outputs_tsne = TSNE(random_state=RS).fit_transform(outputs_pca50)
        np.save(f'./Figures/{sub_dir}/{stage}_outputs_tsne.npy', outputs_tsne)
    # kmeans = KMeans(n_clusters=5, random_state=0).fit(outputs_tsne)
    pred = KMeans(n_clusters=5, random_state=0).fit_predict(outputs_tsne)
    # Slicing to find the most frequent element, map then to the true clusters
    if stage == 'val':
        cluster_map = {np.argmax(np.bincount(pred[i * 200:(i+1) * 200])): i for i in range(5)}
    else:
        cluster_map = {np.argmax(np.bincount(pred[i * 1800:(i + 1) * 1800])): i for i in range(5)}
    pred = np.array([cluster_map[ele] for ele in pred])
    np.save(f'./Figures/{sub_dir}/{stage}_clustering.npy', pred)
    print(f'The R-sqared value of {stage} is: ', r2_score(targets, pred))


# Semi-supervised Labeling
def label_spreading(outputs, targets, num_labels, stage, sub_dir):
    def rbf_kernel_safe(X, Y=None, gamma=None):
        X, Y = sklearn.metrics.pairwise.check_pairwise_arrays(X, Y)
        if gamma is None:
            gamma = 1.0 / X.shape[1]

        K = sklearn.metrics.pairwise.euclidean_distances(X, Y, squared=True)
        K *= -gamma
        K -= K.max()
        np.exp(K, K)  # exponentiate K in-place
        return K

    RS = 42
    rng = np.random.RandomState(RS)
    label_prop_model = LabelSpreading(kernel='rbf')

    if os.path.exists(f'./Figures/{sub_dir}/{stage}_outputs_tsne.npy'):
        outputs_tsne = np.load(f'./Figures/{sub_dir}/{stage}_outputs_tsne.npy')
    else:
        outputs_pca50 = PCA(n_components=50).fit_transform(outputs)
        outputs_tsne = TSNE(random_state=RS).fit_transform(outputs_pca50)
        np.save(f'./Figures/{sub_dir}/{stage}_outputs_tsne.npy', outputs_tsne)

    # outputs_tsne = outputs
    labels = np.copy(targets)

    if stage == 'train':
        idx = [rng.choice(1800, 1800 - num_labels, replace=False) + i * 1800 for i in range(5)]
    else:
        idx = [rng.choice(200, 200 - num_labels, replace=False) + i * 200 for i in range(5)]
    random_unlabeled_points = np.concatenate(idx, axis=0)
    labels[random_unlabeled_points] = -1

    label_prop_model.fit(outputs_tsne, labels)
    pred = label_prop_model.predict(outputs_tsne)
    np.save(f'./Figures/{sub_dir}/{stage}_ls_pred.npy', pred)
    print(f'The label spreading score of {stage} is: ', label_prop_model.score(outputs_tsne, targets))


# Semi-supervised Labeling
def label_propagation(outputs, targets, num_labels, stage, sub_dir):
    def rbf_kernel_safe(X, Y=None, gamma=None):
        X, Y = sklearn.metrics.pairwise.check_pairwise_arrays(X, Y)
        if gamma is None:
            gamma = 1.0 / X.shape[1]

        K = sklearn.metrics.pairwise.euclidean_distances(X, Y, squared=True)
        K *= -gamma
        K -= K.max()
        np.exp(K, K)  # exponentiate K in-place
        return K

    RS = 42
    rng = np.random.RandomState(RS)
    label_prop_model = LabelPropagation(kernel='rbf', tol=0.01, gamma=20)

    if os.path.exists(f'./Figures/{sub_dir}/{stage}_outputs_tsne.npy'):
        outputs_tsne = np.load(f'./Figures/{sub_dir}/{stage}_outputs_tsne.npy')
    else:
        outputs_pca50 = PCA(n_components=50).fit_transform(outputs)
        outputs_tsne = TSNE(random_state=RS).fit_transform(outputs_pca50)
        np.save(f'./Figures/{sub_dir}/{stage}_outputs_tsne.npy', outputs_tsne)

    # outputs_tsne = outputs
    labels = np.copy(targets)

    if stage == 'train':
        idx = [rng.choice(1800, 1800 - num_labels, replace=False) + i * 1800 for i in range(5)]
    else:
        idx = [rng.choice(200, 200 - num_labels, replace=False) + i * 200 for i in range(5)]
    random_unlabeled_points = np.concatenate(idx, axis=0)
    labels[random_unlabeled_points] = -1

    label_prop_model.fit(outputs_tsne, labels)
    pred = label_prop_model.predict(outputs_tsne)
    np.save(f'./Figures/{sub_dir}/{stage}_lp_pred.npy', pred)
    print(f'The label propagation score of {stage} is: ', label_prop_model.score(outputs_tsne, targets))


if __name__ == '__main__':
    main()
