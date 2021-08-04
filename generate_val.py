import os
import shutil
import natsort
import random


def foo(folder='subtomogram_mrc'):
    train_dir = f'./Datasets/train/{folder}'
    val_dir = f'./Datasets/val/{folder}'
    all_files = os.listdir(train_dir)
    sorted_files = natsort.natsorted(all_files)
    # print(sorted_files)
    for i in range(10):
        # print(sorted_files[i*500:(i+1)*500])
        one_class_files = sorted_files[i*500:(i + 1)*500]
        val_files = random.sample(one_class_files, 50)
        # if i == 1: print(val_files)
        for val_file in val_files:
            source_path = os.path.join(train_dir, val_file)
            destination_path = os.path.join(val_dir, val_file)
            shutil.move(source_path, destination_path)
        print(f'class {i} of {folder} is done')


random.seed(3)
foo(folder='subtomogram_mrc')
foo(folder='json_label')
