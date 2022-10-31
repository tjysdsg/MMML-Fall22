import json
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--preds', type=str, default=r'vlp/result/baseline_predictions_val.json')
    parser.add_argument('--labels', type=str, default=r'E:\webqa\data\WebQA_train_val.json')
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.preds, 'r', encoding='utf-8') as f:
        preds = json.load(f)

    num_sources = {}
    for q, data in preds.items():
        sources = data['sources']
        n = len(sources)
        num_sources.setdefault(n, 0)
        num_sources[n] += 1

        if n == 0:
            print(q)

    print({k: num_sources[k] for k in sorted(num_sources)})
    print(63 + 31 + 23 + 11 + 2 + 4 + 1 + 2)


if __name__ == '__main__':
    main()
