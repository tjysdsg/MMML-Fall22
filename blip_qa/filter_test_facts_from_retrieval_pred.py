"""
Create a new WebQA_test.json containing only gold facts predicted by the retriever.
"""
import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=r'E:\webqa\data\WebQA_test.json')
    parser.add_argument('--retrieval-results', type=str, required=True)
    parser.add_argument('--output', type=str, default='test.json')
    return parser.parse_args()


def get_all_facts_for_question(data_q: dict):
    text_facts, img_facts = {}, {}
    for tf in data_q['txt_Facts']:
        text_facts[tf['snippet_id']] = tf
    for tf in data_q['img_Facts']:
        img_facts[tf['image_id']] = tf

    return text_facts, img_facts


def main():
    args = get_args()

    with open(args.data, encoding='utf-8') as f:
        data = json.load(f)
    with open(args.retrieval_results, encoding='utf-8') as f:
        retr = json.load(f)

    res = {}
    for q, v in retr.items():
        q_data = data[q]
        all_text_facts, all_img_facts = get_all_facts_for_question(q_data)
        text_facts, img_facts = [], []
        for s in v['sources']:  # type: str
            if isinstance(s, int):
                img_facts.append(all_img_facts[s])
            else:
                text_facts.append(all_text_facts[s])

        q_data['img_Facts'] = img_facts[:5]
        q_data['txt_Facts'] = text_facts[:5]
        n_facts = len(q_data['img_Facts']) + len(q_data['txt_Facts'])
        if n_facts == 0:
            continue

        res[q] = q_data

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
