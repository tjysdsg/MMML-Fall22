import jsonlines
import json
from collections import defaultdict

with jsonlines.open('./data/WebQA_full_data/val.jsonl', 'r') as f:
    dataset = [obj for obj in f]

gth_dict = {}
for data in dataset:
    if data['label'] == 1:
        qid = data['Q_id']
        if qid not in gth_dict.keys():
            gth_dict[qid] = {'source': [], 'answer': ''}
        if 'txt_fact' in data.keys():
            gth_dict[qid]['source'].append(data['txt_fact']['snippet_id'])
        if 'img_fact' in data.keys():
            gth_dict[qid]['source'].append(data['img_fact']['image_id'])

with open('./data/WebQA_test_data/valid_gth.json', 'w') as f:
    json.dump(gth_dict, f)