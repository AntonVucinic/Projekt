import torch
from torchvision import transforms
from torchvision.ops import box_iou

from data.custom_celeba import CelebA, xywh2xyxy
from models.my_net import MyNet

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

target_transform = transforms.Compose([
    xywh2xyxy,
    torch.Tensor.float
])

valid = CelebA(
    'CelebA',
    'valid',
    transform=transform,
    target_transform=target_transform,
    resize=True
)

model = MyNet()
model.load_state_dict(torch.load('models/model.pth'))
model.eval()


correct = 0
with torch.no_grad():
    for img, target in valid:
        pred = model(img.unsqueeze_(0))
        correct += box_iou(pred, target.unsqueeze_(0)).item() > 0.4
        
print(f'{correct=}')
        