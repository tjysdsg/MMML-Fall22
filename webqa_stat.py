import json
import evaluate
from operator import ge
from collections import defaultdict

def get_dataset():
    with open('./WebQA_data_first_release/WebQA_train_val.json') as f:
        train_dev_dataset = json.load(f)

    with open('./WebQA_data_first_release/WebQA_test.json') as f:
        test_dataset = json.load(f)
    return train_dev_dataset, test_dataset


def get_dataset_num(train_dev_dataset, test_dataset):
    train_num = 0
    val_num = 0
    test_num = 0
    for id, qa_data in train_dev_dataset.items():
        if qa_data['split'] == 'train':
            train_num += 1
        if qa_data['split'] == 'val':
            val_num += 1

    for id, qa_data in test_dataset.items():
        if qa_data['split'] == 'test':
            test_num += 1
    return train_num, val_num, test_num


def get_dataset_neg_pos(dataset, split):
    img_pos_tol = 0
    img_neg_tol = 0
    txt_pos_tol = 0
    txt_neg_tol = 0
    for id, qa_data in dataset.items():
        if qa_data['split'] == split:
            img_pos_tol += len(qa_data['img_posFacts'])
            img_neg_tol += len(qa_data['img_negFacts'])
            txt_pos_tol += len(qa_data['txt_posFacts'])
            txt_neg_tol += len(qa_data['txt_negFacts'])
    return img_neg_tol / img_pos_tol, txt_neg_tol / txt_pos_tol


def get_dataset_topic(dataset, split):
    topic_count = defaultdict(int)
    for id, qa_data in dataset.items():
        if qa_data['split'] == split:
            topic_count[qa_data['topic']] += 1
    return topic_count


def get_topic_overlap(train_topic_count, val_topic_count):
    train_topic = set(train_topic_count.keys())
    val_topic = set(val_topic_count.keys())
    shared_topic = train_topic.intersection(val_topic)
    train_unique_topic = train_topic - shared_topic
    val_unique_topic = val_topic - shared_topic

    unique_val_topic_samples = 0 
    unique_train_topic_samples = 0
    for topic in val_unique_topic:
        unique_val_topic_samples += val_topic_count[topic]
    
    for topic in train_unique_topic:
        unique_train_topic_samples += train_topic_count[topic]

    return len(shared_topic), len(train_unique_topic), len(val_unique_topic), unique_train_topic_samples, unique_val_topic_samples


def get_dataset_qcate(dataset, split):
    qcate_count = defaultdict(int)
    for id, qa_data in dataset.items():
        if qa_data['split'] == split:
            qcate_count[qa_data['Qcate']] += 1
    return qcate_count


def evaluate_bleu(dataset, split, qcate):
    bleu = evaluate.load("bleu")
    for id, qa_data in dataset.items():
        if qa_data['split'] == split and qa_data['Qcate'] == qcate:
            references = [qa_data['Q']]
            answer = qa_data['A']
            print(answer)
            print(references)
            results = bleu.compute(predictions=answer, references=references)
            print(results)
            import pdb; pdb.set_trace()


if __name__ == '__main__':
    train_dev_dataset, test_dataset = get_dataset()
    train_num, val_num, test_num = get_dataset_num(train_dev_dataset, test_dataset)
    train_img_neg_pos_ratio, train_txt_neg_pos_ratio = get_dataset_neg_pos(train_dev_dataset, 'train')
    val_img_neg_pos_ratio, val_txt_neg_pos_ratio = get_dataset_neg_pos(train_dev_dataset, 'val')
    train_topic_count = get_dataset_topic(train_dev_dataset, 'train')
    val_topic_count = get_dataset_topic(train_dev_dataset, 'val')
    shared_topic, train_unique_topic, val_unique_topic, unique_train_topic_samples, unique_val_topic_samples = get_topic_overlap(train_topic_count, val_topic_count)
    train_qcate_count = get_dataset_qcate(train_dev_dataset, 'train')
    val_qcate_count = get_dataset_qcate(train_dev_dataset, 'val')
    evaluate_bleu(train_dev_dataset, 'train', 'YesNo')
    import pdb; pdb.set_trace()