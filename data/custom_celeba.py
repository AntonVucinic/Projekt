import csv
import os
from collections import namedtuple
from functools import partial
from typing import Any, Callable, Literal, Tuple

import PIL
import torch
from torchvision.datasets import VisionDataset

CSV = namedtuple("CSV", ["header", "index", "data"])


class CelebA(VisionDataset):
    def __init__(
            self,
            root: str,
            split: Literal['train', 'test', 'valid', 'all'] = 'train',
            target_type: list[str] | str = 'bbox',
            transform: Callable | None = None,
            target_transform: Callable | None = None,
            size: int | None = None,
            resize: bool = False
    ) -> None:
        super(CelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError(
                'target_transform is specified but target_type is empty')

        split_map = {
            'train': 0,
            'valid': 1,
            'test': 2,
            'all': None,
        }
        split_ = split_map[split.lower()]
        splits = self._load_csv('Eval/list_eval_partition.txt')
        # identity = self._load_csv('Anno/identity_CelebA.txt')
        bbox = self._load_csv('Anno/list_bbox_celeba.txt', header=1)
        # landmarks_align = self._load_csv(
        #     'Anno/list_landmarks_align_celeba.txt', header=1)
        # attr = self._load_csv('Anno/list_attr_celeba.txt', header=1)

        mask = slice(None) if split_ is None else (
            splits.data == split_).squeeze()

        if mask == slice(None):
            self.filename = splits.index
        else:
            self.filename = [splits.index[i]
                             for i in torch.squeeze(torch.nonzero(mask))]
        # self.identity = identity.data[mask]
        self.bbox = bbox.data[mask]
        # self.landmarks_align = landmarks_align.data[mask]
        # self.attr = attr.data[mask]
        # self.attr = torch.div(self.attr + 1, 2, rounding_mode='floor')
        # self.attr_names = attr.header

        self.size = size
        self.resize = resize

    def _load_csv(
        self,
        filename: str,
        header: int | None = None,
    ) -> CSV:
        data, indices, headers = [], [], []

        fn = partial(os.path.join, self.root)
        with open(fn(filename)) as csv_file:
            data = list(csv.reader(
                csv_file, delimiter=' ', skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(
            self.root, "Img/img_celeba", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == 'attr':
                target.append(self.attr[index, :])
            elif t == 'identity':
                target.append(self.identity[index, 0])
            elif t == 'bbox':
                target.append(self.bbox[index, :])
            elif t == 'landmarks':
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError(f'Target type "{t}" is not recognized.')

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)

                if self.resize:
                    w, h = X.size
                    target[0::2] *= 256 / w
                    target[1::2] *= 256 / h

        else:
            target = None

        if self.transform is not None:
            X = self.transform(X)

        return X, target

    def __len__(self) -> int:
        if self.size is not None:
            return self.size
        return len(self.bbox)


def xywh2xyxy(t: torch.Tensor):
    x, y, w, h = t
    return torch.tensor((x, y, x+w, y+h))
