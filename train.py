from pathlib import Path
from typing import Iterable, Optional, Union, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose,
    Resize,
    Normalize
)
from tqdm import tqdm

from data_cleaner.models.vae import AutoEncoder


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_dataset: Iterable,
            val_dataset: Optional[Iterable] = None,
            **kwargs
    ):
        self.model = model
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.lr = kwargs.get('lr', 0.001)
        self.epochs = kwargs.get('epochs', 100)
        self.device = kwargs.get('device', torch.device('cuda'))
        self.criterion = nn.MSELoss()
        self.optimizer = self.init_optimizer(**kwargs)

    def init_optimizer(self, **kwargs):
        optimizer = kwargs.get('optimizer', 'RAdam')
        optimizer = getattr(torch.optim, optimizer)
        optimizer = optimizer(self.model.parameters(), **{'lr': self.lr})
        return optimizer

    def train_loop(self, epoch: int = 0):
        self.model.train()
        avg_loss = 0
        pbar = tqdm(self.train_ds)
        for batch, x in enumerate(pbar):
            self.optimizer.zero_grad()
            x = x.to(self.device)
            output = self.model(x)
            loss = self.criterion(output, x)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()
            pbar.set_description(
                f'[{epoch}/{self.epochs}][{0}/{len(self.train_ds)}]'
                f'Loss = {loss.item()}, Avg Loss = {avg_loss / (batch + 1)}'
            )
        return avg_loss / len(self.train_ds)

    def val_loop(self, epoch: int = 0):
        self.model.eval()
        avg_loss = 0
        pbar = tqdm(self.val_ds)
        for batch, x in enumerate(pbar):
            x = x.to(self.device)
            output = self.model(x)
            loss = self.criterion(output, x).item()
            avg_loss += loss
            pbar.set_description(
                f'[{epoch}/{self.epochs}][{0}/{len(self.val_ds)}]'
                f'Loss = {loss}, Avg Loss = {avg_loss / (batch + 1)}'
            )
        return avg_loss / len(self.val_ds)

    def run(self):
        avg_val_loss = -1
        for epoch in range(self.epochs):
            avg_train_loss = self.train_loop(epoch)
            if self.val_ds is not None:
                avg_val_loss = self.val_loop(epoch)
            print(
                f'Train epoch loss = {avg_train_loss:.2f}, '
                f'Val epoch loss = {avg_val_loss:.2f}'
            )

            # self.model.eval()
            # with torch.no_grad():
            #     indices = range(100, 1000, 100)
            #     for idx in indices:
            #         x = self.val_ds.dataset[idx].to(self.device)
            #         output = self.model(x.view(1, *x.shape))
            #         output = (output.cpu().squeeze().numpy() * 255).astype(np.uint8)
            #         cv2.imshow('output', output)
            #         cv2.waitKey()

class ModernDataset(Dataset):
    def __init__(
            self,
            path: Union[Path, str] = 'MNIST_TRAIN',
            transforms: Callable = None
    ):
        super().__init__()
        if path == 'MNIST_TRAIN':
            self.dataset = datasets.MNIST(
                'data/datasets', train=True, download=True
            )
        elif path == 'MNIST_VAL':
            self.dataset = datasets.MNIST(
                'data/datasets', train=False, download=True
            )
        else:
            self.dataset = datasets.ImageFolder(path)
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = np.array(self.dataset[idx][0])
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        return image


def get_train_dataset(
        batch_size: int = 32,
        target_size: int = 64,
        workers: int = 0
):
    transforms = Compose([
        Resize(target_size, target_size),
        Normalize(0, 1),
        ToTensorV2()
    ])

    train_ds = ModernDataset('MNIST_TRAIN', transforms)
    data_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    return data_loader


def get_val_dataset(
        batch_size: int = 1,
        target_size: int = 64,
        workers: int = 0
):
    transforms = Compose([
        Resize(target_size, target_size),
        Normalize(0, 1),
        ToTensorV2()
    ])

    val_ds = ModernDataset('MNIST_VAL', transforms)
    data_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    return data_loader


def main():
    model = AutoEncoder(
        encoder_name='resnet18', in_channels=1, activation=nn.Sigmoid, z_dim=2
    )
    model.cuda()

    train_ds = get_train_dataset(workers=8)
    val_ds = get_val_dataset(workers=8)
    trainer = Trainer(model, train_dataset=train_ds, val_dataset=val_ds)
    trainer.run()


if __name__ == '__main__':
    main()
