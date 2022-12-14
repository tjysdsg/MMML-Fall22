"""
Change the feature id in the submission json file back to image id
"""

import os
import json
import pickle
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--imgid-map', type=str, required=True)
    parser.add_argument('output', type=str)
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.imgid_map, 'rb') as f:
        imgid_map = pickle.load(f)

        feat2img = {}
        for k, v in imgid_map.items():
            feat2img[v] = k

    with open(args.input) as f:
        res = json.load(f)

    for guid, data in res.items():
        data['sources'] = [feat2img[int(s)] if isinstance(s, int) or s.isnumeric() else s for s in data['sources']]

    with open(args.output, 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
