from argparse import ArgumentParser
import numpy as np
from data_analysis.rcnn_feats import RcnnFeatureLoader
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from PIL import Image
import os

mpl.rcParams['figure.dpi'] = 200


def get_args():
    parser = ArgumentParser()
    parser.add_argument('image', type=int)
    parser.add_argument('--data-dir', type=str, default='/ocean/projects/cis210027p/shared/corpora/webqa',
                        help='Path to the folder containing RCNN feature files')
    parser.add_argument('--imgid-map', type=str,
                        default='/ocean/projects/cis210027p/shared/corpora/webqa/image_id_map_0328.pkl',
                        help='Path to image id map file')
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                        help='Minimum confidence threshold of RCNN bounding boxes')
    return parser.parse_args()


def main():
    args = get_args()

    feat_loader = RcnnFeatureLoader(args.imgid_map, args.data_dir)
    _, class_feats, boxes, scores = [t.cpu().numpy() for t in feat_loader.load_img_feats(args.image)]

    # FIXME: tmp
    with Image.open(os.path.join('tmp', f'{args.image}.jpg')) as im:
        img = np.asarray(im)

    # draw image
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(img)

    # draw bounding boxes
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # only draw boxes that have a score higher than the threshold
        # if scores[i] < args.confidence_threshold:
        #     continue
        if scores[i] > 0.3:
            continue

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y2 - 10, f'score={scores[i]:.2f}', c='r')

    # FIXME: tmp
    plt.savefig(os.path.join('tmp', f'{args.image}_viz.jpg'))
    plt.close('all')

    print(f'class_feats:\n{class_feats}')
    print(f'boxes:\n{boxes}')
    print(f'scores:\n{scores}')


if __name__ == '__main__':
    main()
