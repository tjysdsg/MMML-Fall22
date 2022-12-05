"""
Create a submission file using the generated answer and retriever's output.
"""
import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval-results', type=str, default='../roberta_best_submission.json')
    parser.add_argument('--generator-results', type=str, default='test_pred.json')
    parser.add_argument('--output', type=str, default='submission.json')
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.retrieval_results, encoding='utf-8') as f:
        retr = json.load(f)
    with open(args.generator_results, encoding='utf-8') as f:
        gen = json.load(f)

    answers = {}
    for g in gen:
        answers[g['question_id']] = g['answer']

    res = {}
    for q, q_data in retr.items():
        if q not in answers:
            print(f'Fuck {q}')
            continue
        q_data['answer'] = answers[q]
        res[q] = q_data

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
