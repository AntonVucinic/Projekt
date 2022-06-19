import sys
from itertools import islice

import torch
from torchvision.ops import box_iou

from data.custom_celeba import CelebA, xywh2xyxy


def main():
    assert len(sys.argv) == 4

    _, celeba_root, input_file, shape = sys.argv

    assert shape in ('xyxy', 'xywh')

    celeba = CelebA(
        celeba_root,
        'valid',
        target_transform=xywh2xyxy
    )

    tp = fp = fn = 0

    with open(input_file) as f:
        iter_ = islice(f, 2, None)

        try:
            from numpy import array, float32, int32

            for img, target in celeba:
                name = img.filename

                line = next(iter_)
                img, pred = line.split('\t')

                assert img == name.split('/')[-1]

                while pred.endswith(',\n'):
                    pred += next(iter_)
                try:
                    pred = eval(pred)
                except TypeError:
                    pred = array([])

                if pred is None or len(pred) == 0:
                    fn += 1
                else:
                    pred = torch.tensor(pred)

                    if pred.ndim == 1:
                        pred.unsqueeze_(0)

                    if shape == 'xywh':
                        pred[:, 2] += pred[:, 0]
                        pred[:, 3] += pred[:, 1]

                    total = len(pred)
                    correct = (box_iou(pred, target.unsqueeze_(0))
                               > 0.4).sum().item()
                    tp += correct
                    fp += total - correct
                    
                    # if (box_iou(pred, target.unsqueeze_(0)) > 0.4).any():
                    #     tp += 1
                    # else:
                    #     fp += 1
        except:
            del array, float32, int32

    print(f'{tp=}\n{fp=}\n{fn=}')


if __name__ == '__main__':
    main()
