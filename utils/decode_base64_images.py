"""
Decode WebQA base64 images stored in imgs.tsv, and save them in a directory
"""

import base64
from argparse import ArgumentParser
import os
from io import BytesIO
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default=r'E:\webqa\imgs.tsv')  # 389750 images
    parser.add_argument('--output', type=str, default=r'E:\webqa\images')
    return parser.parse_args()


def main():
    args = get_args()

    input_file = args.input
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as inf:
        for line in inf:
            lineidx, img_base64 = line.rstrip('\n').split()

            try:
                outf = os.path.join(out_dir, f'{lineidx}.jpg')
                if not os.path.exists(outf):  # skip existing images
                    img = Image.open(BytesIO(base64.b64decode(img_base64)))
                    img.convert('RGB').save(outf)
            except (OSError, ValueError) as e:
                print(f'Cannot decode image {lineidx}: {e}')


if __name__ == '__main__':
    main()
