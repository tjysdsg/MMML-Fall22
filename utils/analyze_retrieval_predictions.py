import json
from argparse import ArgumentParser
import numpy as np


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--preds', type=str, default=r'vlp/result/baseline_predictions_val.json')
    parser.add_argument('--labels', type=str, default=r'E:\webqa\data\WebQA_train_val.json')
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.preds, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    with open(args.labels, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    num_sources = {}
    full_correct_q = 0
    overshoot_q = 0
    undershoot_q = 0
    correct_per_qcate = {}
    n_incorrect = 0
    n_multi_source = 0
    correct_multi_source = 0
    n_single_source = 0
    correct_single_source = 0
    for q, data in preds.items():
        sources = [str(s) for s in data['sources']]
        n = len(sources)
        num_sources.setdefault(n, 0)
        num_sources[n] += 1

        # if n > 3:
        #     print(q)
        meta = labels[q]

        qcate = meta['Qcate']
        correct_per_qcate.setdefault(qcate, [])
        true_sources = [str(f['image_id']) for f in meta['img_posFacts']]
        n_true = len(true_sources)

        if n == n_true:  # same number of sources
            if set(sources) == set(true_sources):  # fully correct
                full_correct_q += 1
                correct_per_qcate[qcate].append(1)

                if n_true >= 2:
                    correct_multi_source += 1
                else:
                    correct_single_source += 1
            else:  # but incorrect
                n_incorrect += 1
                correct_per_qcate[qcate].append(0)
        else:  # different number of sources
            n_incorrect += 1
            correct_per_qcate[qcate].append(0)
            if n > n_true:
                overshoot_q += 1
            elif n < n_true:
                undershoot_q += 1

        if n_true >= 2:
            n_multi_source += 1
        else:
            n_single_source += 1

    n_questions = len(preds)
    print('# of sources', {k: num_sources[k] for k in sorted(num_sources)})
    print(f'Full correct questions: {full_correct_q / n_questions}')
    print(f'Undershoot questions: {undershoot_q / n_questions}')
    print(f'Overshoot questions: {overshoot_q / n_questions}')

    print(f'Correct out of multi source: {correct_multi_source / n_multi_source}')
    print(f'Correct out of single source: {correct_single_source / n_single_source}')

    correct_per_qcate = {k: np.mean(v) for k, v in correct_per_qcate.items()}
    print(f'correct per cate: {correct_per_qcate}')


if __name__ == '__main__':
    main()
