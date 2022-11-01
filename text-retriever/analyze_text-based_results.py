import json
from argparse import ArgumentParser
import numpy as np
import evaluate


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--preds', type=str, default=r'./data/WebQA_test_data/submission_text-based_val.json')
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


    f1_metric = evaluate.load('f1')
    recall_metric = evaluate.load('recall')
    precision_metric = evaluate.load('precision')

    results = f1_metric.compute(predictions=pred_labels, references=gth_labels)
    print('F1 for text-based : {}'.format(results['f1']))
    results = recall_metric.compute(predictions=pred_labels, references=gth_labels)    
    print('Recall for text-based : {}'.format(results['recall']))
    results = precision_metric.compute(predictions=pred_labels, references=gth_labels)
    print('Precision for text-based : {}'.format(results['precision']))


if __name__ == '__main__':
    main()