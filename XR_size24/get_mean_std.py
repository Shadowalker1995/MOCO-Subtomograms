#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FileName:	get_mean_std.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-08-04 18:51:36
"""

import os
import numpy as np
import pickle

import Custom_CryoET_DataLoader


class Dataloader:
    def __init__(self):
        self.dirs = ['train', 'val']
        self.means = [0]
        self.stds = [0]

        self.dataset = {stage: Custom_CryoET_DataLoader.CryoETDatasetLoader(
            filename='10_2000_30_01.pickle', stage=stage) for stage in self.dirs}

    def get_mean_std(self, type, mean_std_path):
        """
        calculate the mean and std of the dataset
        :param type: datatype, such as 'train', 'val'
        :param mean_std_path: saved path
        :return:
        """
        num_imgs = len(self.dataset[type])
        for data in self.dataset[type]:
            array = data[0]
            self.means[0] += array[0, :, :, :].mean()
            self.stds[0] += array[0, :, :, :].std()

        self.means = np.asarray(self.means) / num_imgs
        self.stds = np.asarray(self.stds) / num_imgs

        print("{} : NormMean = {}".format(type, self.means))
        print("{} : NormStds = {}".format(type, self.stds))

        with open(mean_std_path, 'wb') as f:
            pickle.dump(self.means, f)
            pickle.dump(self.stds, f)
            print('pickle done')


if __name__ == "__main__":
    root_path = "./Datasets/"
    dataloader = Dataloader()
    for x in dataloader.dirs:
        mean_std_path = root_path + 'mean_std_value_' + x + '.pkl'
        dataloader.get_mean_std(x, mean_std_path)
