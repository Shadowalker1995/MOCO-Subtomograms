# https://discuss.pytorch.org/t/how-to-load-images-without-using-imagefolder/59999/2
# https://github.com/xulabs/projects/blob/master/autoencoder/autoencoder_util.py
import numpy as np
from torch.utils.data import Dataset
import pickle
import itertools
import random


random.seed(42)


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k + min(i, m):(i+1)*k + min(i, m)] for i in list(range(n))]


class CryoETDatasetLoader(Dataset):
    def __init__(self, filename, stage='train', ratio=0.9, transform=None):
        self.filename = filename
        self.transform = transform

        label_list = [b'1I6V', b'1QO1', b'3DY4', b'4V4A', b'5LQW']
        with open(f'./Datasets/{filename}', 'rb') as pickle_file:
            dict_list = pickle.load(pickle_file, encoding='bytes')
        array_list = [np.expand_dims(dict_list[_][b'v'], 0) for _ in range(0, len(dict_list), 1)
                      if dict_list[_][b'id'] in label_list]
        target_list = np.repeat(range(5), 2000)     # ground truth labels
        index_list = list(range(len(array_list)))
        index_split = split(index_list, len(label_list))
        train_index = [random.sample(i, int(2000*ratio)) for i in index_split]
        train_index = list(itertools.chain.from_iterable(train_index))
        if stage == 'train':
            self.arrays = np.array([ele for i, ele in enumerate(array_list) if i in train_index])
            self.targets = np.array([ele for i, ele in enumerate(target_list) if i in train_index])
        elif stage == 'val':
            self.arrays = np.array([ele for i, ele in enumerate(array_list) if i not in train_index])
            self.targets = np.array([ele for i, ele in enumerate(target_list) if i not in train_index])

        print(f'{len(self.arrays)}, vs {len(self.targets)}')
        assert (len(self.arrays) == len(self.targets))

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        array = self.arrays[idx]
        target = self.targets[idx]

        if self.transform is not None:
            transformed_array = self.transform(array)
        else:
            transformed_array = array

        return transformed_array, target


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


if __name__ == "__main__":
    train_loader = CryoETDatasetLoader(filename='10_2000_30_01.pickle', stage='train')
    print(len(train_loader))
    print(train_loader[100][0].shape)

    val_loader = CryoETDatasetLoader(filename='10_2000_30_01.pickle', stage='val')
    print(len(val_loader))
    print(val_loader[100][0].shape)
