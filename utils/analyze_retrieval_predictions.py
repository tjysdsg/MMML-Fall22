import json
from argparse import ArgumentParser
import numpy as np
import evaluate


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--preds', type=str, default=r'vlp/result/baseline_predictions_val.json')
    parser.add_argument('--labels', type=str, default=r'E:\webqa\data\WebQA_train_val.json')
    parser.add_argument('--split', type=str, default='val')
    return parser.parse_args()


class RetrievalStatsCollector:
    def __init__(self, n_questions: int, split='train|val'):
        self.n_questions = n_questions
        self.question_source_count_freq = {}
        self.full_correct_q = 0
        self.n_questoins_with_too_many_predicted_sources = 0
        self.n_questions_with_too_few_predicted_sources = 0
        self.correct_per_qcate = {}
        self.n_incorrect = 0
        self.n_multi_source = 0
        self.correct_multi_source = 0
        self.n_single_source = 0
        self.correct_single_source = 0

        self.gth_labels = []
        self.pred_labels = []

        self.split = split

    def update_f1(self, positive_sources, negative_sources, pred_sources):
        self.gth_labels += [1 for _ in range(len(positive_sources))] + [0 for _ in range(len(negative_sources))]
        for source in positive_sources:
            if source in pred_sources:
                self.pred_labels.append(1)
            else:
                self.pred_labels.append(0)
        for source in negative_sources:
            if source in pred_sources:
                self.pred_labels.append(1)
            else:
                self.pred_labels.append(0)
        assert len(self.gth_labels) == len(self.pred_labels)

    def update(self, pred_sources, question_meta):
        if question_meta['split'] not in self.split:
            return

        pred_sources = [str(p) for p in pred_sources]
        n_pred = len(pred_sources)

        positive_sources = [str(f['image_id']) for f in question_meta['img_posFacts']]
        positive_sources += [str(f['snippet_id']) for f in question_meta['txt_posFacts']]
        negative_sources = [str(f['image_id']) for f in question_meta['img_negFacts']]
        negative_sources += [str(f['snippet_id']) for f in question_meta['txt_negFacts']]

        self.update_f1(positive_sources, negative_sources, pred_sources)

        n_true = len(positive_sources)
        self.question_source_count_freq.setdefault(n_pred, 0)
        self.question_source_count_freq[n_pred] += 1

        qcate = question_meta['Qcate']
        self.correct_per_qcate.setdefault(qcate, [])

        # statistics
        if n_pred == n_true:  # same number of sources
            if set(pred_sources) == set(positive_sources):  # fully correct
                self.full_correct_q += 1
                self.correct_per_qcate[qcate].append(1)

                if n_true >= 2:
                    self.correct_multi_source += 1
                else:
                    self.correct_single_source += 1
            else:  # but incorrect
                self.n_incorrect += 1
                self.correct_per_qcate[qcate].append(0)
        else:  # different number of sources
            self.n_incorrect += 1
            self.correct_per_qcate[qcate].append(0)
            if n_pred > n_true:
                self.n_questoins_with_too_many_predicted_sources += 1
            elif n_pred < n_true:
                self.n_questions_with_too_few_predicted_sources += 1

        if n_true >= 2:
            self.n_multi_source += 1
        else:
            self.n_single_source += 1

    def summary(self):
        print(
            'Number of sources per question:',
            {k: self.question_source_count_freq[k] for k in sorted(self.question_source_count_freq)}
        )
        print(f'Full correct questions: {self.full_correct_q / self.n_questions}')
        print(
            f'Number of questions with too many predicted sources: '
            f'{self.n_questions_with_too_few_predicted_sources / self.n_questions}')
        print(
            f'Number of questions with too few predicted sources: '
            f'{self.n_questoins_with_too_many_predicted_sources / self.n_questions}')

        print(
            f'% of fully correct questions in multi source questions: '
            f'{self.correct_multi_source / self.n_multi_source}'
        )
        print(
            f'% of fully correct in single source questions: '
            f'{self.correct_single_source / self.n_single_source}'
        )

        correct_per_qcate = {k: np.mean(v) for k, v in self.correct_per_qcate.items()}
        print(f'Correct per cate: {correct_per_qcate}')

        # classification evaluation metrics
        f1_metric = evaluate.load('f1')
        recall_metric = evaluate.load('recall')
        precision_metric = evaluate.load('precision')
        results = f1_metric.compute(predictions=self.pred_labels, references=self.gth_labels)
        print(f'F1: {results["f1"]}')
        results = recall_metric.compute(predictions=self.pred_labels, references=self.gth_labels)
        print(f'Recall: {results["recall"]}')
        results = precision_metric.compute(predictions=self.pred_labels, references=self.gth_labels)
        print(f'Precision: {results["precision"]}')


def main():
    args = get_args()

    with open(args.preds, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    with open(args.labels, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    stats = RetrievalStatsCollector(len(preds), split=args.split)

    for q, data in preds.items():
        stats.update(data['sources'], labels[q])

    stats.summary()


if __name__ == '__main__':
    main()
