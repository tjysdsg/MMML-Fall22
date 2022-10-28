"""
Create a small subset from the WebQA data json file

Not actually a random subset, but it can be used to test model training code

Excluding text-only questions
"""

import os
import json
import pickle
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='/ocean/projects/cis210027p/shared/corpora/webqa/WebQA_train_val.json')
    parser.add_argument('--imgid-map', type=str,
                        default='/ocean/projects/cis210027p/shared/corpora/webqa/image_id_map_0328.pkl')
    parser.add_argument('--feat-dir', type=str, default='/ocean/projects/cis210027p/shared/corpora/webqa')
    parser.add_argument('--output', type=str, default='data/webqa_subset.json')
    parser.add_argument('--train-size', type=int, default=1000)  # 1k questions
    parser.add_argument('--val-size', type=int, default=1000)
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.input, "r") as f:
        dataset = json.load(f)

    with open(args.imgid_map, 'rb') as f:
        imgid_map = pickle.load(f)

    subset = {}

    train_size = 0
    val_size = 0
    for i, datum in dataset.items():
        split = datum['split']

        if split == 'train' and train_size >= args.train_size:
            break
        elif split == 'val' and val_size >= args.val_size:
            break

        if datum['Qcate'] == 'text':
            continue

        # check if image features exists
        img_exist = True
        for im in datum['img_posFacts']:
            image_id = int(im['image_id'])
            image_id = imgid_map[image_id]

            image_feature_path = os.path.join(args.feat_dir, f"{split}/{image_id}.pkl")
            if os.path.exists(image_feature_path):
                img_exist = False
                break

        for im in datum['img_negFacts']:
            image_id = int(im['image_id'])
            image_id = imgid_map[image_id]

            image_feature_path = os.path.join(args.feat_dir, f"{split}/{image_id}.pkl")
            if os.path.exists(image_feature_path):
                img_exist = False
                break

        if img_exist:
            subset[i] = datum
            if split == 'train':
                train_size += 1
            elif split == 'val':
                val_size += 1

    with open(args.output, 'w') as f:
        json.dump(subset, f, indent='  ')


if __name__ == '__main__':
    main()
