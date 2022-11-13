import json
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default=r'E:\webqa\data\WebQA_train_val.json')
    parser.add_argument('--output', type=str, default='subWebqa/train_val_multihop.json')
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)

    res = {}
    for q, data in data.items():
        n_pos = len(data['img_posFacts']) + len(data['txt_posFacts'])
        if n_pos >= 2:
            res[q] = data

    print(len(res))

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(res, f)


if __name__ == '__main__':
    main()
