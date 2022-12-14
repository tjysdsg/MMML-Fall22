import json
import random
import jsonlines

def generate_dataset_from_raw_WebQA(file_name, split):
    with open(file_name, 'r') as f:
        raw_dataset = json.load(f)
        dataset = []
        for data_id, data in raw_dataset.items():
            if data['split'] == split:
                Q_id = data['Guid']
                Qcate = data['Qcate']
                Q = data['Q'].replace('"', "")
                A = data['A'][0].replace('"', "")
                topic = data['topic']
                Qcate = data['Qcate']
                pos_txt_facts = []
                neg_txt_facts = []
                pos_img_facts = []
                neg_img_facts = []
                for txt_fact_type in ['txt_posFacts', 'txt_negFacts']:
                    for txt_fact in data[txt_fact_type]:
                        fact = {}
                        fact['title'] = txt_fact['title']
                        fact['fact'] = txt_fact['fact']
                        if txt_fact_type == 'txt_posFacts':
                            fact['label'] = 1
                            pos_txt_facts.append(fact)
                        else:
                            fact['label'] = 0
                            neg_txt_facts.append(fact)
                for img_fact_type in ['img_posFacts', 'img_negFacts']:
                    for img_fact in data[img_fact_type]:
                        fact = {}
                        fact['title'] = img_fact['title']
                        fact['caption'] = img_fact['caption']
                        if img_fact_type == 'img_posFacts':
                            fact['label'] = 1
                            pos_img_facts.append(fact)
                        else:
                            fact['label'] = 0
                            neg_img_facts.append(fact)
                dataset.append({
                    'Q_id': Q_id,
                    'Q': Q, 
                    'A': A, 
                    'topic': topic,
                    'Qcate': Qcate,
                    'pos_img_facts': pos_img_facts, 
                    'pos_txt_facts': pos_txt_facts,
                    'neg_img_facts': neg_img_facts,
                    'neg_txt_facts': neg_txt_facts,
                })
    return dataset

def generate_test_dataset_from_raw_WebQA(file_name):
    with open(file_name, 'r') as f:
        raw_dataset = json.load(f)
        dataset = []
        for data_id, data in raw_dataset.items():
            txt_facts = []
            img_facts = []
            Q_id = data_id
            Q = data['Q'].replace('"', "")
            A = data['A'][0].replace('"', "")
            for txt_fact in data['txt_Facts']:
                fact = {}
                fact['title'] = txt_fact['title']
                fact['fact'] = txt_fact['fact']
                fact['snippet_id'] = txt_fact['snippet_id']
                txt_facts.append(fact)
            for img_fact in data['img_Facts']:
                fact = {}
                fact['title'] = img_fact['title']
                fact['caption'] = img_fact['caption']
                fact['image_id'] = img_fact['image_id']
                img_facts.append(fact)
            dataset.append({
                'Q_id': Q_id, 
                'Q': Q, 
                'A': A, 
                'img_facts': img_facts, 
                'txt_facts': txt_facts
            })
    return dataset


def write_dataset(dataset, output_file_name):
    with jsonlines.open(output_file_name, mode='w') as writer:
        writer.write_all(dataset)
    return


if __name__ == '__main__':
    # for deliberately split small-size WebQA subdata
    file_name = '../text-retriever/raw_data/WebQA_subdata/train_subWebqa.json'
    train_dataset = generate_dataset_from_raw_WebQA(file_name, split='train')
    file_name = '../text-retriever/raw_data/WebQA_subdata/val_subWebqa.json'
    val_dataset = generate_dataset_from_raw_WebQA(file_name, split='val')
    write_dataset(train_dataset, './data/WebQA_sub_data/train.jsonl')
    write_dataset(val_dataset, './data/WebQA_sub_data/val.jsonl')

    # for full-size WebQA data
    file_name = '../text-retriever/raw_data/WebQA_data_first_release/WebQA_train_val.json'
    train_dataset = generate_dataset_from_raw_WebQA(file_name, split='train')
    val_dataset = generate_dataset_from_raw_WebQA(file_name, split='val')
    write_dataset(train_dataset, './data/WebQA_full_data/train.jsonl')
    write_dataset(val_dataset, './data/WebQA_full_data/val.jsonl')

    # for full-size WebQA test data
    file_name = '../text-retriever/raw_data/WebQA_data_first_release/WebQA_test.json'
    test_dataset = generate_test_dataset_from_raw_WebQA(file_name)
    write_dataset(test_dataset, './data/WebQA_test_data/test.jsonl')