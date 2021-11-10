"""
Functions to load the data.

Options
-------
MNIST: Details can be reached from
    https://pytorch.org/docs/stable/torchvision/datasets.html#mnist
Colon Cancer: Details can be reached from
    http://www.warwick.ac.uk/BIAlab/data/CRChistoLabeledNucleiHE
"""
# Import modules
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from tqdm import tqdm
import numpy as np
import argparse
import torch
import copy
import os

# Import own modules
from dataloaders.load_warwick import load_warwick


def load_mnist(args: argparse.Namespace) -> (
        MNIST, MNIST
):
    """
    Load the MNIST dataset.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments of the process.

    Returns
    -------
    return1: MNIST
        Train loader.
    return2: MNIST
        Test loader.
    """
    # Set transform
    if args.model == 'base_att':
        # Trace
        print('Loading MNIST. Normalization is active.')
        # With normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        # Trace
        print('Loading MNIST. Bernoulli Transformation is active.')
        # With Bernoulli
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.bernoulli(x))
        ])

    # Return train/test sets without onehot target transform
    return MNIST(root='./data',
                 train=True,
                 transform=transform,
                 download=True), MNIST(root='./data',
                                       train=False,
                                       transform=transform,
                                       download=True)


def create_semisupervised_datasets_for_MNIST(dataset: MNIST,
                                             n_labeled: int) -> (MNIST, MNIST):
    """
    Divide the data as labeled and unlabeled.

    Parameters
    ----------
    dataset: torchvision.datasets.mnist.MNIST
        The MNIST dataset to be divided.
    n_labeled: int
        The number of instances will be in the labeled set.

    Returns
    -------
    return1: MNIST
        The labeled MNIST set.
    return2: MNIST
        The unlabeled MNIST set.
    """
    # note this is only relevant for training the model
    assert dataset.train is True, \
        'Dataset must be the training set; assure dataset.train = True.'

    # Compile new x and y and replace the dataset.train_data and train_labels with the
    x = dataset.data
    y = dataset.targets
    # indices = torch.randperm(6000)
    # x, y = x[indices], y[indices]
    n_classes = len(torch.unique(y))

    assert n_labeled % n_classes == 0, \
        'n_labeld not divisible by n_classes; cannot assure class balance.'
    n_labeled_per_class = n_labeled // n_classes

    x_labeled = [0] * n_classes
    x_unlabeled = [0] * n_classes
    x_validation = [0] * n_classes
    y_labeled = [0] * n_classes
    y_unlabeled = [0] * n_classes
    y_validation = [0] * n_classes

    for i in range(n_classes):
        idxs = torch.nonzero((y == i), as_tuple=False).data.numpy()
        np.random.shuffle(idxs)

        x_labeled[i] = x[idxs][:n_labeled_per_class]
        y_labeled[i] = y[idxs][:n_labeled_per_class]
        x_validation[i] = x[idxs][n_labeled_per_class:n_labeled_per_class + 500]
        y_validation[i] = y[idxs][n_labeled_per_class:n_labeled_per_class + 500]
        x_unlabeled[i] = x[idxs][n_labeled_per_class + 500:]
        y_unlabeled[i] = y[idxs][n_labeled_per_class + 500:]

    # construct new labeled and unlabeled datasets
    labeled_dataset = copy.deepcopy(dataset)
    labeled_dataset.data = torch.cat(x_labeled, dim=0).squeeze()
    labeled_dataset.targets = torch.cat(y_labeled, dim=0)

    unlabeled_dataset = copy.deepcopy(dataset)
    unlabeled_dataset.data = torch.cat(x_unlabeled, dim=0).squeeze()
    unlabeled_dataset.targets = torch.cat(y_unlabeled, dim=0)

    # Construct the validation set
    validation_dataset = copy.deepcopy(dataset)
    validation_dataset.data = torch.cat(x_validation, dim=0).squeeze()
    validation_dataset.targets = torch.cat(y_validation, dim=0)

    del dataset

    return labeled_dataset, unlabeled_dataset, validation_dataset


def create_semisupervised_datasets_for_Colon(dataset,
                                             n_labeled: int):
    """
    Divide the data as labeled and unlabeled.

    Parameters
    ----------
    dataset: torchvision.datasets.mnist.MNIST
        The MNIST dataset to be divided.
    n_labeled: int
        The number of instances will be in the labeled set.

    Returns
    -------
    return1: MNIST
        The labeled MNIST set.
    return2: MNIST
        The unlabeled MNIST set.
    """
    # Extract the bag of data from the dataset
    x = [data[0] for data in dataset]
    # Extract the targets from the dataset
    y = [data[1][0] for data in dataset]
    # Declare the number of classes
    n_classes = len(torch.unique(y))

    assert n_labeled % n_classes == 0, \
        'n_labeld not divisible by n_classes; cannot assure class balance.'
    n_labeled_per_class = n_labeled // n_classes

    x_labeled = [0] * n_classes
    x_unlabeled = [0] * n_classes
    x_validation = [0] * n_classes
    y_labeled = [0] * n_classes
    y_unlabeled = [0] * n_classes
    y_validation = [0] * n_classes

    for i in range(n_classes):
        idxs = torch.nonzero((y == i), as_tuple=False).data.numpy()
        np.random.shuffle(idxs)

        x_labeled[i] = x[idxs][:n_labeled_per_class]
        y_labeled[i] = y[idxs][:n_labeled_per_class]
        x_validation[i] = x[idxs][n_labeled_per_class:n_labeled_per_class + 500]
        y_validation[i] = y[idxs][n_labeled_per_class:n_labeled_per_class + 500]
        x_unlabeled[i] = x[idxs][n_labeled_per_class + 500:]
        y_unlabeled[i] = y[idxs][n_labeled_per_class + 500:]

    # construct new labeled and unlabeled datasets
    labeled_dataset = copy.deepcopy(dataset)
    labeled_dataset.data = torch.cat(x_labeled, dim=0).squeeze()
    labeled_dataset.targets = torch.cat(y_labeled, dim=0)

    unlabeled_dataset = copy.deepcopy(dataset)
    unlabeled_dataset.data = torch.cat(x_unlabeled, dim=0).squeeze()
    unlabeled_dataset.targets = torch.cat(y_unlabeled, dim=0)

    # Construct the validation set
    validation_dataset = copy.deepcopy(dataset)
    validation_dataset.data = torch.cat(x_validation, dim=0).squeeze()
    validation_dataset.targets = torch.cat(y_validation, dim=0)

    del dataset

    return labeled_dataset, unlabeled_dataset, validation_dataset


class MnistBags(Dataset):
    def __init__(self,
                 dataset,
                 target_number=9,
                 mean_bag_length=10,
                 var_bag_length=0,
                 seed=1,
                 train=True):
        self.dataset = dataset
        self.lendataset = len(dataset)
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = self.lendataset // self.mean_bag_length
        self.train = train

        self.r = np.random.RandomState(seed)

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        loader = DataLoader(self.dataset,
                            batch_size=self.lendataset,
                            shuffle=True) if self.train \
            else DataLoader(self.dataset,
                            batch_size=self.lendataset,
                            shuffle=False)

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1

            if self.train:
                indices = torch.LongTensor(self.r.randint(0, self.lendataset, bag_length))
            else:
                indices = torch.LongTensor(self.r.randint(0, self.lendataset, bag_length))

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label


def semiAttMILMNIST(args: argparse.Namespace,
                    trainset: MNIST,
                    testset: MNIST) -> (DataLoader, DataLoader, DataLoader):
    # Divide trainset for semi-supervised training
    labeledset, unlabeledset, validationset = create_semisupervised_datasets_for_MNIST(
        trainset, args.n_labeled
    )
    # Create loaders
    for data in ['train', 'labeled', 'unlabeled', 'validation', 'test']:
        globals()[f'{data}loader'] = DataLoader(MnistBags(dataset=locals()[f'{data}set'],
                                                          train=True)) \
            if data == 'labeled' or data == 'unlabeled' \
            else DataLoader(MnistBags(dataset=locals()[f'{data}set'],
                                      train=False))

    # Return loaders
    return trainloader, labeledloader, unlabeledloader, validationloader, testloader


def kfold_indices_warwick(N, k, seed=777):
    r = np.random.RandomState(seed)
    all_indices = np.arange(N, dtype=int)
    r.shuffle(all_indices)
    idx = [int(i) for i in np.floor(np.linspace(0, N, k + 1))]
    train_folds = []
    valid_folds = []
    for fold in range(k):
        valid_indices = all_indices[idx[fold]:idx[fold + 1]]
        valid_folds.append(valid_indices)
        train_fold = np.setdiff1d(all_indices, valid_indices)
        r.shuffle(train_fold)
        train_folds.append(train_fold)
    return train_folds, valid_folds


def load_colonCancer(args: argparse.Namespace):
    # Declare the training and the test folds [indices]
    train_folds, test_folds = kfold_indices_warwick(100, 10, seed=args.seed)
    # Declare the training and the validation folds [indices]
    train_fold, val_fold = kfold_indices_warwick(len(train_folds[0]), 10, seed=args.seed)
    train_fold = [train_folds[0][i] for i in train_fold][0]
    val_fold = [train_folds[0][i] for i in val_fold][0]
    # Import the train, the validation, and the test sets
    train_set, val_set, test_set = load_warwick(train_fold,
                                                val_fold,
                                                test_folds[0],
                                                padding=args.padding,
                                                base_att=True if args.model == 'base_att' else False)
    # Return sets
    return train_set, val_set, test_set


def semi_supervised_setup_for_Colon(trainset, n_labeled):
    # Declare empty lists
    labeled, unlabeled = list(), list()
    # Declare the tracker of the number of the labeled set
    labeled_cnt = 0
    # Declare trackers for un/labeled instances in the bags
    labeled_instances_cnt, unlabeled_instances_cnt = 0, 0
    # Declare the indices to be picked while constructing the labeled set
    indices = np.arange(0, len(trainset), 1).tolist()
    # indices = np.arange(0, 10, 1).tolist()
    # Loop through the trainset
    for _ in tqdm(range(len(trainset)), desc='Un/labeled separations'):
        # Pick a random indice
        idx = np.random.choice(indices)
        # Declare the instance according to the indice
        instance = trainset[idx]
        # Remove that indice from the indices list
        indices.remove(idx)
        # If the number of labeled instances is below the given length
        # and equals to the randomly picked label tag
        if labeled_cnt < n_labeled and instance[1][0] == np.random.choice([0.0, 1.0]):
            # Append the instance to the labeled set list
            labeled.append(instance)
            # Increase the number of labeled bag counter
            labeled_cnt += 1
            # Increase the number of labeled instance in a bag counter
            labeled_instances_cnt += instance[0].shape[0]
        # If the number of labeled data is higher than 60
        elif n_labeled > 60:
            # Once more the same condition as above statement
            # to be able fill enough number of labeled data
            if labeled_cnt < n_labeled and instance[1][0] == np.random.choice([0.0, 1.0]):
                # Append the instance to the labeled set list
                labeled.append(instance)
                # Increase the number of labeled bag counter
                labeled_cnt += 1
                # Increase the number of labeled instance in a bag counter
                labeled_instances_cnt += instance[0].shape[0]
            # Otherwise
            else:
                # Append the instance to the unlabeled set list
                unlabeled.append(instance)
                # Increase the number of unlabeled instance in a bag counter
                unlabeled_instances_cnt += instance[0].shape[0]
        # Otherwise
        else:
            # Append the instance to the unlabeled set list
            unlabeled.append(instance)
            # Increase the number of unlabeled instance in a bag counter
            unlabeled_instances_cnt += instance[0].shape[0]

    # Delete unneeded variables
    del labeled_cnt
    del labeled_instances_cnt
    del unlabeled_instances_cnt
    del instance
    # Return the sets
    return labeled, unlabeled


def labelDistOverBags(labeled):
    # Declare an empty dictionary
    dist = dict()
    # Loop through the labeled set
    for data in labeled:
        # Declare the label
        label = data[1][0]
        # If the label is not in the dictionary
        if label not in dist:
            # Create it
            dist[label] = 0
        # Increase the number by one
        dist[label] += 1
    # Return the distribution records
    return dist


def semiAttMILColon(args: argparse.Namespace) -> (DataLoader,
                                                  DataLoader,
                                                  DataLoader,
                                                  DataLoader,
                                                  DataLoader):
    # Load the Colon Cancer sets
    trainset, validationset, testset = load_colonCancer(args=args)
    # Trace
    print('Colon Cancer set is loaded.')
    # Divide the trainset into labeled, and unlabeled sets
    labeledset, unlabeledset = semi_supervised_setup_for_Colon(trainset, args.n_labeled)
    # Trace
    print('Semi-supervised setup is ready.',
          f'#Labeled: {len(labeledset)}, #Unlabeled: {len(unlabeledset)}.',
          f'#Validation: {len(validationset)}, #Test: {len(testset)}.')
    # Record the label distribution within the labeled bag
    dist = labelDistOverBags(labeledset)
    # Trace
    print('The distribution of the label over the bags;',
          f'#True: {dist[1.0]}, #False: {dist[0.0]}')

    # Create loaders
    for data in ['train', 'labeled', 'unlabeled', 'validation', 'test']:
        globals()[f'{data}loader'] = DataLoader(locals()[f'{data}set'], batch_size=args.bs, shuffle=True) \
            if data != 'test' \
            else DataLoader(locals()[f'{data}set'], batch_size=args.bs, shuffle=False)
        # Save the loader
        isNorm = True if args.model == 'base_att' else False
        torch.save(globals()[f'{data}loader'],
                   f'{args.LOADERPATH}/{data}loader_{args.data}_{args.padding}_{args.n_labeled}_isNorm{isNorm}.pth')
    # Return loaders
    return trainloader, labeledloader, unlabeledloader, validationloader, testloader


def load_data(args: argparse.Namespace):
    """
    Load data according to the given model type.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments of the process.

    Returns
    -------
    return1: DataLoader
        The train set loader.
    return2: DataLoader
        The test set loader.
    """
    # If data MNIST
    if args.data == 'MNIST':
        trainset, testset = load_mnist(args=args)
        trainloader, labeledloader, unlabeledloader, \
            validationloader, testloader = semiAttMILMNIST(args=args,
                                                           trainset=trainset,
                                                           testset=testset)
        # Trace
        print('Semi-supervised Attention MIL MNIST set is loaded.')
        # Return semi-supervised Attention MIL MNIST
        return trainloader, labeledloader, unlabeledloader, \
            validationloader, testloader
    # If the data is the Colon Cancer dataset
    elif args.data == 'Colon':
        # Declare whether the dataset is normalized for base_att
        isNorm = True if args.model == 'base_att' else False
        # Declare a loadername [trainloader]
        loadername = f'{args.LOADERPATH}/trainloader_{args.data}_{args.padding}_{args.n_labeled}_isNorm{isNorm}.pth'
        # If the loaders wasn't saved earlier
        if not os.path.exists(loadername):
            # Declare loaders
            trainloader, labeledloader, unlabeledloader, \
                validationloader, testloader = semiAttMILColon(args=args)
        # If the loaders was saved
        else:
            loaders = list()
            # Load the DataLoaders
            for data in ['train', 'labeled', 'unlabeled', 'validation', 'test']:
                # Update the loadername
                loadername = f'{args.LOADERPATH}/{data}loader_{args.data}_{args.padding}_{args.n_labeled}_isNorm{isNorm}.pth'
                loaders.append(torch.load(loadername))
            del loadername
            return loaders
        # Return the loaders
        return trainloader, labeledloader, unlabeledloader, \
            validationloader, testloader
