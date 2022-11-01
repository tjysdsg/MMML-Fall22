import json
from argparse import ArgumentParser
import numpy as np
import evaluate


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--preds', type=str, default=r'./data/WebQA_test_data/submission_image-based_val.json')
    parser.add_argument('--labels', type=str, default=r'./raw_data/WebQA_data_first_release/WebQA_train_val.json')
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

    gth_labels = []
    pred_labels = []

    for q, data in preds.items():
        sources = data['sources']
        n = len(sources)
        num_sources.setdefault(n, 0)
        num_sources[n] += 1

        # if n > 3:
        #     print(q)
        meta = labels[q]

        qcate = meta['Qcate']
        correct_per_qcate.setdefault(qcate, [])
        true_sources = [f['image_id'] for f in meta['img_posFacts']]
        true_sources += [f['snippet_id'] for f in meta['txt_posFacts']]
        false_sources = [f['image_id'] for f in meta['img_negFacts']]
        false_sources += [f['snippet_id'] for f in meta['txt_negFacts']]

        gth_labels += [1 for _ in range(len(true_sources))] + [0 for _ in range(len(false_sources))]
        for source in true_sources:
            if source in data['sources']:
                pred_labels.append(1)
            else:
                pred_labels.append(0)
        for source in false_sources:
            if source in data['sources']:
                pred_labels.append(1)
            else:
                pred_labels.append(0)
        assert len(gth_labels) == len(pred_labels)

        #for img_id in true_sources:
        #    if img_id

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

    f1_metric = evaluate.load('f1')
    recall_metric = evaluate.load('recall')
    precision_metric = evaluate.load('precision')

    results = f1_metric.compute(predictions=pred_labels, references=gth_labels)
    print('F1 for image-based : {}'.format(results['f1']))
    results = recall_metric.compute(predictions=pred_labels, references=gth_labels)    
    print('Recall for image-based : {}'.format(results['recall']))
    results = precision_metric.compute(predictions=pred_labels, references=gth_labels)
    print('Precision for image-based : {}'.format(results['precision']))

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