import json
import random
import jsonlines

def generate_dataset_from_raw_WebQA(file_name):
    with open(file_name, 'r') as f:
        dataset = json.load(f)
        val_dataset = []
        train_dataset = []
        for data_id, data in dataset.items():
            Qcate = data['Qcate']
            Q = data['Q'].replace('"', "")
            A = data['A'][0].replace('"', "")
            txt_facts = []
            for pos_fact in data['txt_posFacts']:
                fact = {}
                fact['title'] = pos_fact['title']
                fact['fact'] = pos_fact['fact']
                fact['label'] = 1
                #txt_facts.append(fact)
                if data['split'] == 'train':
                    train_dataset.append({'Q': Q, 'A': A, 'txt_facts': fact})
                elif data['split'] == 'val':
                    val_dataset.append({'Q': Q, 'A': A, 'txt_facts': fact})
            for neg_fact in data['txt_negFacts']:
                fact = {}
                fact['title'] = neg_fact['title']
                fact['fact'] = neg_fact['fact']
                fact['label'] = 0
                #txt_facts.append(fact)
                if data['split'] == 'train':
                    train_dataset.append({'Q': Q, 'A': A, 'txt_facts': fact})
                elif data['split'] == 'val':
                    val_dataset.append({'Q': Q, 'A': A, 'txt_facts': fact})
    random.shuffle(train_dataset)
    random.shuffle(val_dataset)
    return train_dataset, val_dataset


def write_dataset(dataset, output_file_name):
    with jsonlines.open(output_file_name, mode='w') as writer:
        writer.write_all(dataset)
    return


if __name__ == '__main__':
    file_name = './raw_data/WebQA_data_first_release/WebQA_train_val.json'
    train_dataset, val_dataset = generate_dataset_from_raw_WebQA(file_name)
    write_dataset(train_dataset, './data/train.jsonl')
    write_dataset(val_dataset, './data/val.jsonl')
