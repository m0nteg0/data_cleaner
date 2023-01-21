from typing import Optional, List, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import \
    UnetDecoder, CenterBlock, DecoderBlock
from segmentation_models_pytorch.base import (
    SegmentationHead
)
from albumentations.pytorch import ToTensorV2


class Decoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
        z_dim: int = 10
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, 0, out_ch, **kwargs)
            for in_ch, out_ch in zip(in_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.linear = nn.Linear(z_dim, 512)

    def forward(self, z, encoder_output_size: int = 7):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, size=encoder_output_size)
        for i, decoder_block in enumerate(self.blocks):
            x = decoder_block(x, None)
        return x


class AutoEncoder(nn.Module):
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
            z_dim: int = 10,
            activation: Optional[Union[str, callable]] = None
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = Decoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
            z_dim=z_dim
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=in_channels,
            activation=activation,
            kernel_size=3,
        )

        self.linear = nn.Linear(512, 2 * z_dim)
        self.z_dim = z_dim

    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def reparameterize(self, x):
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mean = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        encoder_output = self.encoder(x)[-1]
        z = self.reparameterize(encoder_output)
        decoder_output = self.decoder(z, encoder_output.shape[-1])
        result = self.segmentation_head(decoder_output)

        return result


def main():
    train_ds = datasets.MNIST('data/datasets', train=True, download=True)
    test_img = np.array(train_ds[1000][0])
    test_img = cv2.resize(test_img, (64, 64))
    model = AutoEncoder(
        encoder_name='resnet18', in_channels=1, activation=nn.Sigmoid, z_dim=2
    )
    model.cuda()
    # model = smp.Unet(
    #     encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    #     in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #     classes=1,  # model output channels (number of classes in your dataset)
    # )

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
