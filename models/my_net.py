import sys

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes


class ConvBlock(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, kernel_size=(3, 3), stride=(2, 2))
        self.bn = nn.BatchNorm2d(out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 512)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(512)
        self.fc = nn.Linear(512, 4)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.bn(x)
        return self.fc(x)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])

    img = Image.open(sys.argv[1])
    img = transform(img)

    model = MyNet().to(device)
    model.load_state_dict(torch.load('models/model.pth'))
    model.eval()

    pred = model(img.unsqueeze_(0).to(device))

    img = (img.squeeze_(0) * 255).to(dtype=torch.uint8)
    pred = pred.to(dtype=torch.uint8)

    img = draw_bounding_boxes(img, pred)
    img = transforms.ToPILImage()(img)
    img.save('out.png', 'png')


if __name__ == '__main__':
    main()
