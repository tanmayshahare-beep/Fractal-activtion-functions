"""
Data Loading Utilities (PyTorch)

This module provides functions to load MNIST and Fashion-MNIST datasets
from local idx files for PyTorch.
"""

import os
import numpy as np
import idx2numpy
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def load_mnist_local(dataset_dir):
    """
    Load MNIST dataset from local idx files.

    Args:
        dataset_dir: Directory containing the idx files

    Returns:
        Tuple of ((train_images, train_labels), (test_images, test_labels))
        Images are numpy arrays of shape (N, 28, 28, 1), labels are (N,)
    """
    train_images = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 'train-images.idx3-ubyte'))
    train_labels = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 'train-labels.idx1-ubyte'))
    test_images = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 't10k-images.idx3-ubyte'))
    test_labels = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 't10k-labels.idx1-ubyte'))

    # Add channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    return (train_images, train_labels), (test_images, test_labels)


def load_fashion_mnist_local(dataset_dir):
    """
    Load Fashion-MNIST dataset from local idx files.

    Args:
        dataset_dir: Directory containing the idx files

    Returns:
        Tuple of ((train_images, train_labels), (test_images, test_labels))
    """
    # Handle different naming conventions
    if os.path.exists(os.path.join(dataset_dir, 'train-images.idx3-ubyte')):
        train_images_path = os.path.join(dataset_dir, 'train-images.idx3-ubyte')
        train_labels_path = os.path.join(dataset_dir, 'train-labels.idx1-ubyte')
        test_images_path = os.path.join(dataset_dir, 't10k-images.idx3-ubyte')
        test_labels_path = os.path.join(dataset_dir, 't10k-labels.idx1-ubyte')
    else:
        train_images_path = os.path.join(dataset_dir, 'train-images-idx3-ubyte')
        train_labels_path = os.path.join(dataset_dir, 'train-labels-idx1-ubyte')
        test_images_path = os.path.join(dataset_dir, 't10k-images-idx3-ubyte')
        test_labels_path = os.path.join(dataset_dir, 't10k-labels-idx1-ubyte')

    train_images = idx2numpy.convert_from_file(train_images_path)
    train_labels = idx2numpy.convert_from_file(train_labels_path)
    test_images = idx2numpy.convert_from_file(test_images_path)
    test_labels = idx2numpy.convert_from_file(test_labels_path)

    # Add channel dimension
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    return (train_images, train_labels), (test_images, test_labels)


class MNISTDataset(Dataset):
    """
    PyTorch Dataset for MNIST from local idx files.

    Args:
        images_path: Path to images idx file
        labels_path: Path to labels idx file
        transform: Optional transforms to apply
        target_transform: Optional target transforms

    Example:
        >>> dataset = MNISTDataset(
        ...     images_path='datasets/MNIST/train-images.idx3-ubyte',
        ...     labels_path='datasets/MNIST/train-labels.idx1-ubyte',
        ...     transform=transforms.ToTensor()
        ... )
    """

    def __init__(self, images_path, labels_path, transform=None, target_transform=None):
        self.images = idx2numpy.convert_from_file(images_path)
        self.labels = idx2numpy.convert_from_file(labels_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to PIL Image for transforms
        image = Image.fromarray(image.squeeze(), mode='L')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class FashionMNISTDataset(Dataset):
    """
    PyTorch Dataset for Fashion-MNIST from local idx files.
    """

    def __init__(self, images_path, labels_path, transform=None, target_transform=None):
        self.images = idx2numpy.convert_from_file(images_path)
        self.labels = idx2numpy.convert_from_file(labels_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to PIL Image
        image = Image.fromarray(image.squeeze(), mode='L')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class NumpyDataset(Dataset):
    """
    PyTorch Dataset from numpy arrays.

    Args:
        images: Numpy array of shape (N, H, W, C) or (N, H, W)
        labels: Numpy array of shape (N,)
        transform: Optional transforms

    Example:
        >>> (x_train, y_train), _ = load_mnist_local('datasets/MNIST')
        >>> dataset = NumpyDataset(x_train, y_train, transform=transforms.ToTensor())
    """

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Handle different image shapes
        if len(image.shape) == 2:
            image = Image.fromarray(image, mode='L')
        elif len(image.shape) == 3:
            if image.shape[-1] == 1:
                image = Image.fromarray(image.squeeze(), mode='L')
            else:
                image = Image.fromarray(image, mode='RGB')
        else:
            image = Image.fromarray(image, mode='L')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_mnist_dataloaders(dataset_dir, batch_size=64, val_split=0.1):
    """
    Create DataLoaders for MNIST dataset.

    Args:
        dataset_dir: Directory containing MNIST idx files
        batch_size: Batch size for DataLoaders
        val_split: Fraction of training data to use for validation

    Returns:
        train_loader, val_loader, test_loader
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist_local(dataset_dir)

    # Split validation from training
    n_train = len(x_train)
    n_val = int(val_split * n_train)
    indices = torch.randperm(n_train)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    # Create datasets
    train_dataset = NumpyDataset(
        x_train[train_indices.numpy()],
        y_train[train_indices.numpy()],
        transform=train_transform)

    val_dataset = NumpyDataset(
        x_train[val_indices.numpy()],
        y_train[val_indices.numpy()],
        transform=test_transform)

    test_dataset = NumpyDataset(x_test, y_test, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_fashion_mnist_dataloaders(dataset_dir, batch_size=64, val_split=0.1):
    """
    Create DataLoaders for Fashion-MNIST dataset.

    Args:
        dataset_dir: Directory containing Fashion-MNIST idx files
        batch_size: Batch size for DataLoaders
        val_split: Fraction of training data to use for validation

    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    # Load data
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist_local(dataset_dir)

    # Split validation from training
    n_train = len(x_train)
    n_val = int(val_split * n_train)
    indices = torch.randperm(n_train)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    # Create datasets
    train_dataset = NumpyDataset(
        x_train[train_indices.numpy()],
        y_train[train_indices.numpy()],
        transform=train_transform)

    val_dataset = NumpyDataset(
        x_train[val_indices.numpy()],
        y_train[val_indices.numpy()],
        transform=test_transform)

    test_dataset = NumpyDataset(x_test, y_test, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
