import sys

import torch
from data.custom_celeba import CelebA
from facenet_pytorch import MTCNN


def main():
    assert len(sys.argv) in (3, 4)

    celeba_root = sys.argv[1]
    output_file = sys.argv[2]
    
    size = int(sys.argv[3]) if len(sys.argv) == 4 else None

    model = MTCNN()

    dataset = CelebA(
        celeba_root,
        split='train',
        target_type=[],
    )

    with open(output_file, 'w') as f:
        f.write(f'{len(dataset)}\n')
        f.write('image_id\tboxes\n')

        model.eval()
        with torch.no_grad():
            for idx, (img, _) in enumerate(dataset):
                pred, _ = model.detect(img)
                
                if pred is not None:
                    pred = pred[0]
                    
                f.write(f'{idx:06d}.jpg\t{pred!r}\n')
                
                if size and idx >= size:
                    break


if __name__ == '__main__':
    main()
