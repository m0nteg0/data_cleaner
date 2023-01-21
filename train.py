from typing import Iterable, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from albumentations.pytorch import ToTensorV2
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
        self.epochs = kwargs.get('epochs', 0.001)
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
            x = x[0] if isinstance(x, tuple) else x
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
            x = x[0] if isinstance(x, tuple) else x
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


def main():
    train_ds = datasets.MNIST('data/datasets', train=True, download=True)
    test_img = np.array(train_ds[1000][0])
    test_img = cv2.resize(test_img, (64, 64))
    model = AutoEncoder(
        encoder_name='resnet18', in_channels=1, activation=nn.Sigmoid, z_dim=2
    )
    model.cuda()

    trainer = Trainer(model, train_dataset=train_ds)
    a = 0

    criterion = nn.MSELoss()
    optimizer = torch.optim.RAdam(model.parameters(), 0.001)

    for i in range(500):
        input_tensor = torch.from_numpy(test_img) / 255
        input_tensor = input_tensor.view(1, 1, *input_tensor.shape).cuda()
        output = model(input_tensor)
        optimizer.zero_grad()
        loss = criterion(output, input_tensor)
        loss.backward()
        optimizer.step()
        print(loss)

    input_tensor = torch.from_numpy(test_img) / 255
    input_tensor = input_tensor.view(1, 1, *input_tensor.shape).cuda()
    output = model(input_tensor).squeeze()
    output = output.detach().cpu().numpy()
    output = (output * 255).astype(np.uint8)

    cv2.imshow('source', test_img)
    cv2.imshow('output', output)
    cv2.waitKey()



    a = 0



if __name__ == '__main__':
    main()
