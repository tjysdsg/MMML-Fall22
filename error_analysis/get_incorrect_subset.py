import json
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--preds', type=str, required=True)
    parser.add_argument('--labels', type=str, default=r'E:\webqa\data\WebQA_train_val.json')
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.preds, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    with open(args.labels, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    res = {}
    for q, data in preds.items():
        pred_sources = [str(p) for p in data['sources']]
        n_pred = len(pred_sources)

        positive_sources = [str(f['image_id']) for f in labels[q]['img_posFacts']]
        positive_sources += [str(f['snippet_id']) for f in labels[q]['txt_posFacts']]
        n_true = len(positive_sources)

        if n_pred == n_true and set(pred_sources) == set(positive_sources):  # fully correct
            continue

        res[q] = labels[q]

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)


if __name__ == '__main__':
    main()
