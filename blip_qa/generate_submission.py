"""
Create a submission file using the generated answer and retriever's output.
"""
import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval-results', type=str, default='../roberta_best_submission.json')
    parser.add_argument('--generator-results', type=str, default='test_pred_uncased_img_only_multitask.json')
    parser.add_argument('--output', type=str, default='test_submission_uncased_img_only_multitask.json')
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.retrieval_results, encoding='utf-8') as f:
        retr = json.load(f)
    with open(args.generator_results, encoding='utf-8') as f:
        gen = json.load(f)

    answers = {}
    for g in gen:
        answers[g['question_id']] = g['pred']

    res = {}
    n_missing = 0
    for q, q_data in retr.items():
        if q not in answers:
            print(f'Missing {q}')
            n_missing += 1
        else:
            q_data['answer'] = answers[q]
        res[q] = q_data

    print(f'n_missing: {n_missing}')

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
