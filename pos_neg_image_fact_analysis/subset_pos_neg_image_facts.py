"""Subset positive and negative images from question facts"""
import json
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--data-json', type=str,
                        default=r'E:\webqa\data\WebQA_train_val.json',
                        help='Path to the data json file')
    parser.add_argument('--train-size', type=int, default=2000,
                        help='Number of questions whose images are used for the train set')
    parser.add_argument('--test-size', type=int, default=1000,
                        help='Number of questions whose images are used for the test set')
    parser.add_argument('--output-train', type=str, default='train.tsv',
                        help='Output file path, format = question_id\timage_id\tpositive/negative')
    parser.add_argument('--output-test', type=str, default='test.tsv',
                        help='Output file path, format = question_id\timage_id\tpositive/negative')
    return parser.parse_args()


def subset_data(facts: list, output_file: str):
    qids = []
    image_ids = []
    labels = []
    n_positive = 0
    for qid, d in facts:
        pos = d['img_posFacts']
        neg = d['img_negFacts']

        for fact in pos:
            qids.append(qid)
            image_ids.append(int(fact['image_id']))
            labels.append(1)
            n_positive += 1

        for fact in neg[:2]:  # balance positive and negative
            qids.append(qid)
            image_ids.append(int(fact['image_id']))
            labels.append(0)

    print(f'Positive samples: {n_positive}/{len(labels)}[{n_positive / len(labels):.2f}]')

    with open(output_file, 'w') as f:
        for i, image_id in enumerate(image_ids):
            f.write(f'{qids[i]}\t{image_id}\t{labels[i]}\n')


def main():
    args = get_args()

    with open(args.data_json) as f:
        data: dict = json.load(f)

    # split facts according to split==train or split==val
    facts = list(data.items())
    train_facts = [(qid, f) for qid, f in facts if f['split'] == 'train']
    test_facts = [(qid, f) for qid, f in facts if f['split'] == 'val']

    print("Creating train set...")
    subset_data(train_facts, args.output_train)

    print("Creating test set...")
    subset_data(test_facts, args.output_test)


if __name__ == '__main__':
    main()
