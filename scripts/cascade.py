import sys

import cv2 as cv
from ..data.custom_celeba import CelebA, xywh2xyxy
from torchvision.transforms import PILToTensor


def main():
    assert len(sys.argv) == 4

    cascade_path = sys.argv[1]
    output_file = sys.argv[2]
    celeba_root = sys.argv[3]

    data = CelebA(
        celeba_root,
        'all',
        transform=PILToTensor(),
        target_transform=xywh2xyxy
    )

    face_cascade = cv.CascadeClassifier()
    face_cascade.load(cascade_path)

    with open(output_file, 'w') as f:
        f.write(f'{len(data)}\n')
        f.write('image_id\tboxes\n')

        for idx, (img, _) in enumerate(data):
            ndarr = img.permute((1, 2, 0)).numpy()
            img_gray = cv.cvtColor(ndarr, cv.COLOR_RGB2GRAY)
            img_gray = cv.equalizeHist(img_gray)

            pred = face_cascade.detectMultiScale(img_gray)
            f.write(f'{idx:06d}.jpg\t{pred!r}\n')


if __name__ == '__main__':
    main()
