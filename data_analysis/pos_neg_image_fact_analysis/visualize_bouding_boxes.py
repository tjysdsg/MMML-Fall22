import os
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from data_analysis.rcnn_feats import RcnnFeatureLoader
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 200


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--image-dir', type=str, default='/ocean/projects/cis210027p/jtang1/webqa/images5000',
                        help='Path to the folder containing images')
    parser.add_argument('--data-dir', type=str, default='/ocean/projects/cis210027p/jtang1/webqa',
                        help='Path to the folder containing RCNN feature files')
    parser.add_argument('--imgid-map', type=str,
                        default='/ocean/projects/cis210027p/jtang1/webqa/image_id_map_0328.pkl',
                        help='Path to image id map file')
    parser.add_argument('-N', type=int, default=10, help='Number of questions whose images are used')
    parser.add_argument('--out-dir', type=str, default='exp', help='Output dir')
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                        help='Minimum confidence threshold of RCNN bounding boxes')
    return parser.parse_args()


class RCNNData:
    def __init__(self, imgid_map: str, image_dir: str, feat_dir: str):
        self.feat_loader = RcnnFeatureLoader(imgid_map, feat_dir)
        self.images = os.listdir(image_dir)
        self.image_dir = image_dir

    def __getitem__(self, index):
        image_name = self.images[index]

        # load image
        img_path = os.path.join(self.image_dir, image_name)
        with Image.open(img_path) as im:
            img = np.asarray(im)

        # load RCNN features
        img_id = int(image_name.split('.')[0])

        try:
            # FIXME: which split
            _, class_feats, boxes, scores = self.feat_loader.load_img_feats(img_id, 'train')
        except FileNotFoundError as e:
            return None

        class_feats, boxes, scores = class_feats.numpy(), boxes.numpy(), scores.numpy()

        return img_id, img, class_feats, boxes, scores

    def __len__(self):
        return len(self.images)


def main():
    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    data = RCNNData(args.imgid_map, args.image_dir, args.data_dir)
    n = 0
    for d in data:
        if n >= args.N:
            break

        if d is None:  # skip those that failed to load
            continue
        img_id, img, class_feats, boxes, scores = d

        # draw image
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(img)

        # draw bounding boxes
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # only draw boxes that have a score higher than the threshold
            if scores[i] < args.confidence_threshold:
                continue

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y2 - 10, f'score={scores[i]:.2f}', c='r')

        # close all figures
        plt.savefig(os.path.join(out_dir, f'{img_id}.jpg'))
        plt.close('all')

        n += 1


if __name__ == '__main__':
    main()
