import sys

import torch
from data.custom_celeba import CelebA
from facenet_pytorch import MTCNN
import numpy as np


def main():
    assert len(sys.argv) == 3

    celeba_root = sys.argv[1]
    output_file = sys.argv[2]
    
    model = MTCNN()

    dataset = CelebA(
        celeba_root,
        split='valid',
        target_type=[],
    )

    with open(output_file, 'w') as f:
        f.write(f'{len(dataset)}\n')
        f.write('image_id\tboxes\n')

        model.eval()
        with torch.no_grad():
            for img, _ in dataset:
                pred, probs = model.detect(img)
                
                if pred is not None:
                    if len(pred) > 1:
                        filter_ = np.array(probs) > 0.99
                        pred = pred[filter_]
                    else:
                        pred, = pred
                    
                f.write(f'{img.filename.split("/")[-1]}\t{pred!r}\n')
                

if __name__ == '__main__':
    main()
