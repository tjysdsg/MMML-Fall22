import jsonlines
import json

def get_retrieved_dataset(retrieved_submit_results, full_test_dataset):
    with open(retrieved_submit_results, 'r') as json_f:
        submit_results = json.load(json_f)

    
    with jsonlines.open(full_test_dataset, 'r') as jsonl_f:
        full_test_dataset = [obj for obj in jsonl_f]

    retrieved_dataset = []
    for test_data in full_test_dataset:
        qid = test_data['Q_id']
        Q = test_data['Q']
        A = test_data['A']
        retrieved_src_ids = submit_results[qid]['sources']
        retrieved_img_facts = []
        retrieved_txt_facts = []
        for img_fact in test_data['img_facts']:
            if img_fact['image_id'] in retrieved_src_ids:
                retrieved_img_facts.append(img_fact)
        for txt_fact in test_data['txt_facts']:
            if txt_fact['snippet_id'] in retrieved_src_ids:
                retrieved_txt_facts.append(txt_fact)
        assert len(retrieved_src_ids) == (len(retrieved_txt_facts) + len(retrieved_img_facts))
        retrieved_dataset.append({
            'Q_id':  qid,
            'Q': Q,
            'A': A,
            'img_facts': retrieved_img_facts,
            'txt_facts': retrieved_txt_facts
        })
    return retrieved_dataset


def save_dataset(dataset, output_file_name):
    with jsonlines.open(output_file_name, mode='w') as writer:
        writer.write_all(dataset)
    return


if __name__ == '__main__':
    retrieved_submit_results = '../text-retriever/data/WebQA_test_data/submission_test.json'
    full_test_dataset = './data/WebQA_test_data/test.jsonl'
    retrieved_dataset = get_retrieved_dataset(retrieved_submit_results, full_test_dataset)
    save_dataset(retrieved_dataset, './data/WebQA_test_data/retrieved_test.jsonl')