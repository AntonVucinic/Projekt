from itertools import islice
import sys

import torch
from torchvision.ops import box_iou


from data.custom_celeba import CelebA, xywh2xyxy


def main():
    assert len(sys.argv) == 4

    _, celeba_root, input_file, shape = sys.argv
    
    assert shape in('xyxy', 'xywh')

    celeba = CelebA(
        celeba_root,
        'all',
        target_transform=xywh2xyxy
    )

    tp = fp = fn = 0

    with open(input_file) as f:
        iter_ = islice(f, 2, None)
        try:
            from numpy import array, int32, float32
            while True:
                line = next(iter_)
                img, pred = line.split('\t')
                while pred.endswith(',\n'):
                    pred += next(iter_)
                pred = eval(pred)
                
                if pred is None or len(pred) == 0:
                    fn += 1
                else:
                    pred = torch.tensor(pred)
                
                    if pred.ndim == 1:
                        pred.unsqueeze_(0)
                    
                    _, target = celeba[int(img[:6])]
                    if shape =='xywh':
                        pred[:, 2] += pred[:, 0]
                        pred[:, 3] += pred[:, 1]
                    pos = (box_iou(pred, target.unsqueeze_(0)) > 0.4).sum()
                    neg = len(pred) - pos
                    
                    tp += pos
                    fp += neg

        except StopIteration:
            del array, int32, float32

    print(f'{tp=}\n{fp=}\n{fn=}')


if __name__ == '__main__':
    main()
