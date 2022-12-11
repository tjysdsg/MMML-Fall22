import json
from typing import List
from webqa_eval import webqa_fl, webqa_acc_approx


def main():
    with open('val_pred.json') as f:
        data = json.load(f)

    preds = []
    refs = []
    qcates = []
    for d in data:
        preds.append(d['pred'])
        refs.append(d['answer'])
        qcates.append(d['qcate'])

    metrics = calc_qa_metrics(preds, refs, qcates)
    metrics['qa'] = metrics['fl'] * metrics['acc']
    print(metrics)


def calc_qa_metrics(preds: List[str], refs: List[str], qcates: List[str]):
    assert len(preds) == len(refs) == len(qcates), f'{len(preds)} {len(refs)} {len(qcates)}'

    # save bart scores
    bart_scores = webqa_fl(preds, refs)
    bart_res = []
    i = 0
    for pred, ref, qcate in zip(preds, refs, qcates):
        bart_res.append(dict(
            pred=pred,
            answer=ref,
            qcate=qcate,
            bart_score=bart_scores['scores'][i],
        ))
        i += 1
    with open('bart_scores.json', 'w', encoding='utf-8') as f:
        json.dump(bart_res, f)

    ret = {'color': [], 'shape': [], 'YesNo': [], 'number': [], 'text': [], 'Others': [], 'choose': [],
           'f1': [], 'recall': [], 'acc': [], 'fl': bart_scores['fl']}
    for pred, ref, qcate in zip(preds, refs, qcates):
        eval_output = webqa_acc_approx(pred, ref, qcate)['acc_approx']
        ret[qcate].append(eval_output)
        if qcate in ['color', 'shape', 'number', 'YesNo']:
            ret['f1'].append(eval_output)
        else:
            ret['recall'].append(eval_output)
        ret['acc'].append(eval_output)

    for key, value in ret.items():
        if key == 'fl':
            continue
        if len(ret[key]) == 0:
            ret[key] = 0
        else:
            ret[key] = sum(ret[key]) / len(ret[key])

    return ret


if __name__ == '__main__':
    main()
