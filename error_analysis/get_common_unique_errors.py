import json
import os
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--vlp', type=str, required=True)
    parser.add_argument('--roberta', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    return parser.parse_args()


def main():
    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    with open(args.vlp, 'r', encoding='utf-8') as f:
        vlp = json.load(f)
    with open(args.roberta, 'r', encoding='utf-8') as f:
        roberta = json.load(f)

    vlp_questions = set(vlp.keys())
    roberta_questions = set(roberta.keys())

    common = {}
    vlp_unique = {}
    roberta_unique = {}
    for k in vlp_questions & roberta_questions:
        common[k] = vlp[k]

    for k in vlp_questions - roberta_questions:
        vlp_unique[k] = vlp[k]
    for k in roberta_questions - vlp_questions:
        roberta_unique[k] = roberta[k]

    with open(os.path.join(out_dir, 'common_error_subset.json'), 'w', encoding='utf-8') as f:
        json.dump(common, f, indent=4)
    with open(os.path.join(out_dir, 'vlp_unique_error_subset.json'), 'w', encoding='utf-8') as f:
        json.dump(vlp_unique, f, indent=4)
    with open(os.path.join(out_dir, 'roberta_unique_error_subset.json'), 'w', encoding='utf-8') as f:
        json.dump(roberta_unique, f, indent=4)


if __name__ == '__main__':
    main()
