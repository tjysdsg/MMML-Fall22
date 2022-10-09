import json
from collections import defaultdict, Counter

def get_dataset(path):
    with open(path) as f:
        dataset = json.load(f)
    return dataset


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


def get_dataset_neg_pos(dataset, split, qa_type):
    img_pos_tol = 0
    img_neg_tol = 0
    txt_pos_tol = 0
    txt_neg_tol = 0
    total_num = 0
    for id, qa_data in dataset.items():
        if qa_data['split'] == split:
            if qa_type == 'text-based':
                if qa_data['img_posFacts'] == []:
                    txt_pos_tol += len(qa_data['txt_posFacts'])
                    txt_neg_tol += len(qa_data['txt_negFacts'])
                    img_pos_tol += len(qa_data['img_posFacts'])
                    img_neg_tol += len(qa_data['img_negFacts'])
                    total_num += 1
                if qa_data['txt_posFacts'] == []:
                    continue
                    #img_pos_tol += len(qa_data['img_posFacts'])
                    #img_neg_tol += len(qa_data['img_negFacts'])
            elif qa_type == 'img-based':
                if qa_data['img_posFacts'] == []:
                    continue
                if qa_data['txt_posFacts'] == []:
                    txt_pos_tol += len(qa_data['txt_posFacts'])
                    txt_neg_tol += len(qa_data['txt_negFacts'])
                    img_pos_tol += len(qa_data['img_posFacts'])
                    img_neg_tol += len(qa_data['img_negFacts'])
                    total_num += 1
            else:
                raise NotImplementedError
    return img_pos_tol/total_num, img_neg_tol/total_num, \
           txt_pos_tol/total_num, txt_neg_tol/total_num


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

    unique_val_topic_sample_num = 0 
    unique_train_topic_sample_num = 0
    for topic in val_unique_topic:
        unique_val_topic_sample_num += val_topic_count[topic]
    
    for topic in train_unique_topic:
        unique_train_topic_sample_num += train_topic_count[topic]

    return len(shared_topic), \
           len(train_unique_topic), len(val_unique_topic), \
           unique_train_topic_sample_num, unique_val_topic_sample_num


def get_dataset_qcate(dataset, split):
    qcate_count = defaultdict(int)
    for id, qa_data in dataset.items():
        if qa_data['split'] == split:
            qcate_count[qa_data['Qcate']] += 1
    return qcate_count


if __name__ == '__main__':
    # get dataset
    train_dev_dataset = get_dataset('./WebQA_data_first_release/WebQA_train_val.json')
    test_dataset = get_dataset('./WebQA_data_first_release/WebQA_test.json')
    train_num, val_num, test_num = get_dataset_num(train_dev_dataset, test_dataset)

    # neg-pos stat
    train_img_neg_pos_stat = get_dataset_neg_pos(train_dev_dataset, 'train', 'img-based')
    train_text_neg_pos_stat = get_dataset_neg_pos(train_dev_dataset, 'train', 'text-based')
    val_img_neg_pos_stat = get_dataset_neg_pos(train_dev_dataset, 'val', 'img-based')
    val_text_neg_pos_stat = get_dataset_neg_pos(train_dev_dataset, 'val', 'text-based')

    # topic stat
    train_topic_count = get_dataset_topic(train_dev_dataset, 'train')
    val_topic_count = get_dataset_topic(train_dev_dataset, 'val')
    shared_topic, train_unique_topic, val_unique_topic, unique_train_topic_sample_num, unique_val_topic_sample_num = get_topic_overlap(train_topic_count, val_topic_count)

    # qcate stat
    train_qcate_count = get_dataset_qcate(train_dev_dataset, 'train')
    val_qcate_count = get_dataset_qcate(train_dev_dataset, 'val')
