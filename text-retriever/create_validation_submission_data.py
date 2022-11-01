import json
import random 
import jsonlines

with open('./raw_data/WebQA_data_first_release/WebQA_train_val.json') as f:
    raw_dataset = json.load(f)
    dataset = []
    for data_id, data in raw_dataset.items():
        if data['split'] == 'val' and data['img_posFacts'] == []:
            Q_id = data_id
            Q = data['Q'].replace('"', "")
            A = data['A'][0].replace('"', "")
            for txt_fact in data['txt_posFacts']:
                fact = {}
                fact['title'] = txt_fact['title']
                fact['fact'] = txt_fact['fact']
                fact['snippet_id'] = txt_fact['snippet_id']
                dataset.append({'Q_id': Q_id, 'Q': Q, 'A': A, 'txt_fact': fact})
            for txt_fact in data['txt_negFacts']:
                fact = {}
                fact['title'] = txt_fact['title']
                fact['fact'] = txt_fact['fact']
                fact['snippet_id'] = txt_fact['snippet_id']
                dataset.append({'Q_id': Q_id, 'Q': Q, 'A': A, 'txt_fact': fact})
            for img_fact in data['img_posFacts']:
                fact = {}
                fact['title'] = img_fact['title']
                fact['caption'] = img_fact['caption']
                fact['image_id'] = img_fact['image_id']
                dataset.append({'Q_id': Q_id, 'Q': Q, 'A': A, 'img_fact': fact})
            for img_fact in data['img_negFacts']:
                fact = {}
                fact['title'] = img_fact['title']
                fact['caption'] = img_fact['caption']
                fact['image_id'] = img_fact['image_id']
                dataset.append({'Q_id': Q_id, 'Q': Q, 'A': A, 'img_fact': fact})

output_file_name = './data/WebQA_test_data/text-based_val_as_test.jsonl'
with jsonlines.open(output_file_name, mode='w') as writer:
    writer.write_all(dataset)
