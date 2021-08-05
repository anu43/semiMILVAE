"""Pytorch Dataset object that loads 500x500 patches. only used for checking instance scores
 not for training."""

import os
import scipy.io
import numpy as np
from PIL import Image
import torch.utils.data as data_utils
import torchvision.transforms as transforms


class ColonCancerWhole(data_utils.Dataset):
    def __init__(self, path, train_val_idxs=None, test_idxs=None, train=True):
        self.path = path
        self.train_val_idxs = train_val_idxs
        self.test_idxs = test_idxs
        self.train = train

        self.to_tensor_transform = transforms.Compose([transforms.ToTensor()])

        self.dir_list_train, self.dir_list_test = self.split_dir_list(
            self.path, self.train_val_idxs, self.test_idxs)
        if self.train:
            self.img_list_train, self.labels_list_train, self.coordinates_train = self.create_bags(
                self.dir_list_train)
        else:
            self.img_list_test, self.labels_list_test, self.coordinates_test = self.create_bags(
                self.dir_list_test)

    @staticmethod
    def split_dir_list(path, train_val_idxs, test_idxs):
        dirs = [x[0] for x in os.walk(path)]
        dirs.pop(0)
        dirs.sort()

        dir_list_train = [dirs[i] for i in train_val_idxs]
        dir_list_test = [dirs[i] for i in test_idxs]

        return dir_list_train, dir_list_test

    @staticmethod
    def create_bags(dir_list):
        img_list = []
        labels_list = []
        coordinate_list = []

        for dir in dir_list:
            # Get image name
            img_name = dir.split('/')[-1]

            # bmp to pillow
            img_dir = dir + '/' + img_name + '.bmp'
            with open(img_dir, 'rb') as f:
                with Image.open(f) as img:
                    img = img.convert('RGB')

            # crop malignant cells
            dir_epithelial = dir + '/' + img_name + '_epithelial.mat'
            with open(dir_epithelial, 'rb') as f:
                mat_epithelial = scipy.io.loadmat(f)

            # crop all other cells
            dir_inflammatory = dir + '/' + img_name + '_inflammatory.mat'
            dir_fibroblast = dir + '/' + img_name + '_fibroblast.mat'
            dir_others = dir + '/' + img_name + '_others.mat'

            with open(dir_inflammatory, 'rb') as f:
                mat_inflammatory = scipy.io.loadmat(f)
            with open(dir_fibroblast, 'rb') as f:
                mat_fibroblast = scipy.io.loadmat(f)
            with open(dir_others, 'rb') as f:
                mat_others = scipy.io.loadmat(f)

            benign_coordinates = np.concatenate((mat_inflammatory['detection'].astype(
                float), mat_fibroblast['detection'].astype(float), mat_others['detection'].astype(float)), axis=0)
            all_coordinates = np.concatenate((mat_epithelial['detection'].astype(float), mat_inflammatory['detection'].astype(
                float), mat_fibroblast['detection'].astype(float), mat_others['detection'].astype(float)), axis=0)

            # store single cell labels
            labels = np.concatenate(
                (np.ones(len(mat_epithelial['detection'])), np.zeros(len(benign_coordinates))), axis=0)

            img_list.append(img)
            labels_list.append(labels)
            coordinate_list.append(all_coordinates)

        return img_list, labels_list, coordinate_list

    def __len__(self):
        if self.train:
            return len(self.labels_list_train)
        else:
            return len(self.labels_list_test)

    def __getitem__(self, index):
        if self.train:
            img = self.to_tensor_transform(self.img_list_train[index])
            label = [max(self.labels_list_train[index]), self.labels_list_train[index]]
            coordinates = self.coordinates_train[index]
        else:
            img = self.to_tensor_transform(self.img_list_test[index])
            label = [max(self.labels_list_test[index]), self.labels_list_test[index]]
            coordinates = self.coordinates_test[index]

        return img, label, coordinates
