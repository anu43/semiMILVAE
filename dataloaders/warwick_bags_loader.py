# """Pytorch Dataset object that loads 27x27 patches that contain single cells."""

import os
import random

import numpy as np
import scipy.io
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.functional import pad

import dataloaders.additional_transforms as AT


class ColonCancerBagsCross(data_utils.Dataset):
    def __init__(
        self,
        path,
        train_val_idxs=None,
        test_idxs=None,
        train=True,
        shuffle_bag=False,
        data_augmentation=False,
        padding=True,
        base_att=False,
    ):
        self.path = path
        self.train_val_idxs = train_val_idxs
        self.test_idxs = test_idxs
        self.train = train
        self.shuffle_bag = shuffle_bag
        self.data_augmentation = data_augmentation
        self.padding = padding
        self.base_att = base_att

        if self.base_att:
            # Trace
            # print('Normalization enabled on the Colon Cancer dataset.')
            self.data_augmentation_img_transform = transforms.Compose(
                [
                    AT.RandomHEStain(),
                    AT.HistoNormalize(),
                    AT.RandomRotate(),
                    AT.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            self.normalize_to_tensor_transform = transforms.Compose(
                [
                    AT.HistoNormalize(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            # Trace
            # print('Normalization disabled on the Colon Cancer dataset.')
            self.data_augmentation_img_transform = transforms.Compose(
                [
                    AT.RandomHEStain(),
                    AT.HistoNormalize(),
                    AT.RandomRotate(),
                    AT.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            self.normalize_to_tensor_transform = transforms.Compose(
                [
                    AT.HistoNormalize(),
                    transforms.ToTensor(),
                ]
            )

        self.dir_list_train, self.dir_list_test = self.split_dir_list(
            self.path, self.train_val_idxs, self.test_idxs
        )
        if self.train:
            self.bag_list_train, self.labels_list_train = self.create_bags(
                self.dir_list_train
            )
        else:
            self.bag_list_test, self.labels_list_test = self.create_bags(
                self.dir_list_test
            )

    @staticmethod
    def split_dir_list(path, train_val_idxs, test_idxs):
        dirs = [x[0] for x in os.walk(path)]
        dirs.pop(0)
        dirs.sort()

        dir_list_train = [dirs[i] for i in train_val_idxs]
        dir_list_test = [dirs[i] for i in test_idxs]

        return dir_list_train, dir_list_test

    def create_bags(self, dir_list):
        bag_list = []
        labels_list = []
        for dir in dir_list:
            # Get image name
            img_name = dir.split("/")[-1]

            # bmp to pillow
            img_dir = dir + "/" + img_name + ".bmp"
            with open(img_dir, "rb") as f:
                with Image.open(f) as img:
                    img = img.convert("RGB")

            # crop malignant cells
            dir_epithelial = dir + "/" + img_name + "_epithelial.mat"
            with open(dir_epithelial, "rb") as f:
                mat_epithelial = scipy.io.loadmat(f)

            cropped_cells_epithelial = []
            for (x, y) in mat_epithelial["detection"]:
                x = np.round(x)
                y = np.round(y)

                if self.data_augmentation:
                    x = x + np.round(np.random.normal(0, 3, 1))
                    y = y + np.round(np.random.normal(0, 3, 1))

                # If it is a numpy array
                if type(x) == np.ndarray:
                    x = x[0]
                if x < 13:
                    x_start = 0
                    x_end = 27
                elif x > 500 - 13:
                    x_start = 500 - 27
                    x_end = 500
                else:
                    x_start = x - 13
                    x_end = x + 14

                # If it is a numpy array
                if type(y) == np.ndarray:
                    y = y[0]
                if y < 13:
                    y_start = 0
                    y_end = 27
                elif y > 500 - 13:
                    y_start = 500 - 27
                    y_end = 500
                else:
                    y_start = y - 13
                    y_end = y + 14

                cropped_cells_epithelial.append(
                    img.crop((x_start, y_start, x_end, y_end))
                )

            # crop all other cells
            dir_inflammatory = dir + "/" + img_name + "_inflammatory.mat"
            dir_fibroblast = dir + "/" + img_name + "_fibroblast.mat"
            dir_others = dir + "/" + img_name + "_others.mat"

            with open(dir_inflammatory, "rb") as f:
                mat_inflammatory = scipy.io.loadmat(f)
            with open(dir_fibroblast, "rb") as f:
                mat_fibroblast = scipy.io.loadmat(f)
            with open(dir_others, "rb") as f:
                mat_others = scipy.io.loadmat(f)

            all_coordinates = np.concatenate(
                (
                    mat_inflammatory["detection"],
                    mat_fibroblast["detection"],
                    mat_others["detection"],
                ),
                axis=0,
            )

            cropped_cells_others = []
            for (x, y) in all_coordinates:
                x = np.round(x)
                y = np.round(y)

                if self.data_augmentation:
                    x = x + np.round(np.random.normal(0, 3, 1))
                    y = y + np.round(np.random.normal(0, 3, 1))

                # If it is a numpy array
                if type(x) == np.ndarray:
                    x = x[0]
                if x < 13:
                    x_start = 0
                    x_end = 27
                elif x > 500 - 13:
                    x_start = 500 - 27
                    x_end = 500
                else:
                    x_start = x - 13
                    x_end = x + 14

                # If it is a numpy array
                if type(y) == np.ndarray:
                    y = y[0]
                if y < 13:
                    y_start = 0
                    y_end = 27
                elif y > 500 - 13:
                    y_start = 500 - 27
                    y_end = 500
                else:
                    y_start = y - 13
                    y_end = y + 14

                cropped_cells_others.append(img.crop((x_start, y_start, x_end, y_end)))

            # generate bag
            bag = cropped_cells_epithelial + cropped_cells_others

            # store single cell labels
            labels = np.concatenate(
                (
                    np.ones(len(cropped_cells_epithelial)),
                    np.zeros(len(cropped_cells_others)),
                ),
                axis=0,
            )

            # shuffle
            if self.shuffle_bag:
                zip_bag_labels = list(zip(bag, labels))
                random.shuffle(zip_bag_labels)
                bag, labels = zip(*zip_bag_labels)

            # append every bag two times if training
            if self.train:
                for _ in [0, 1]:
                    bag_list.append(bag)
                    labels_list.append(labels)
            else:
                bag_list.append(bag)
                labels_list.append(labels)

            # bag_list.append(bag)
            # labels_list.append(labels)

        return bag_list, labels_list

    def transform_and_data_augmentation(self, bag):
        if self.data_augmentation:
            img_transform = self.data_augmentation_img_transform
        else:
            img_transform = self.normalize_to_tensor_transform

        bag_tensors = []
        for img in bag:
            # If padding is True
            if self.padding:
                bag_tensors.append(
                    pad(img_transform(img), (0, 1, 0, 1), mode="constant")
                )
            # Otherwise
            else:
                bag_tensors.append(img_transform(img))
        return torch.stack(bag_tensors)

    def __len__(self):
        if self.train:
            return len(self.labels_list_train)
        else:
            return len(self.labels_list_test)

    def __getitem__(self, index):
        if self.train:
            bag = self.bag_list_train[index]
            label = [max(self.labels_list_train[index]), self.labels_list_train[index]]
        else:
            bag = self.bag_list_test[index]
            label = [max(self.labels_list_test[index]), self.labels_list_test[index]]

        return self.transform_and_data_augmentation(bag), label
