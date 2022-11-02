import os
import jsonlines
import argparse
import torch
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class WebQATestDataset(Dataset):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.data = self.build_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def recursive_tokenize(self, data):
        if isinstance(data, int):
            return data
        elif isinstance(data, str):
            return self.tokenizer(data.strip(), truncation=True, add_special_tokens=False)['input_ids']
        elif isinstance(data, dict):
            for key, value in data.items():
                if key == 'image_id' or key == 'snippet_id' or key == 'Q_id':
                    data[key] = value
                else:
                    data[key] = self.recursive_tokenize(value)

            return data
        
        lists = list(self.recursive_tokenize(subdata) for subdata in data)
        return lists    

    def build_dataset(self):
        if self.args.have_cached_dataset:
            dataset = torch.load(os.path.join(self.args.cache_dir, 'WebQA_test_dataset'))
        else:
            with jsonlines.open(os.path.join(self.args.dataset_dir, self.args.test_file), 'r') as jsonl_f:
                dataset = [obj for obj in jsonl_f]
            
            print('=====begin tokenize======')
            dataset = self.recursive_tokenize(dataset)
            print('=====end   tokenize======')
            torch.save(dataset, os.path.join(self.args.cache_dir, 'WebQA_test_dataset'))
        return dataset

    def collate_fn(self, batch, max_length=None):
        input_ids = []
        labels = []
        attention_mask = []
        source_types = []
        q_ids = []
        source_ids = []

        for instance in batch:
            instance_token_ids = [self.tokenizer.cls_token_id]
            # QA part
            instance_token_ids += instance['Q']
            #instance_token_ids += instance['A'] # answer is all empty in test cases
            # add one [SEP] after QA part
            instance_token_ids += [self.tokenizer.sep_token_id]
            # fact part (both for text and image)
            if 'txt_fact' in instance.keys():
                instance_token_ids += instance['txt_fact']['fact']
                source_types.append('txt')
                source_ids.append(instance['txt_fact']['snippet_id'])
            elif 'img_fact' in instance.keys():
                instance_token_ids += instance['img_fact']['caption']
                source_types.append('img')
                source_ids.append(instance['img_fact']['image_id'])
            else:
                raise ValueError('instance should either be image-based or text-based')

            instance_token_ids = instance_token_ids[:self.args.max_length-1] # since there is one last [SEP]
            # add [SEP] after truncation
            instance_token_ids += [self.tokenizer.sep_token_id]
            instance_token_ids = torch.LongTensor(instance_token_ids)

            input_ids.append(instance_token_ids)
            q_ids.append(instance['Q_id'])


        input_ids = pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)


        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'source_ids': source_ids,
            'source_types': source_types,
            'q_ids': q_ids,
        }



class WebQADataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.tokenizer = tokenizer
        self.data = self.build_dataset(split=split)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def recursive_tokenize(self, data):
        if isinstance(data, int):
            return data
        elif isinstance(data, str):
            return self.tokenizer(data.strip(), truncation=True, add_special_tokens=False)['input_ids']
        elif isinstance(data, dict):
            return dict((key, self.recursive_tokenize(value)) for key, value in data.items())
        
        lists = list(self.recursive_tokenize(subdata) for subdata in data)
        return lists    

    def build_dataset(self, split):
        if self.args.have_cached_dataset:
            dataset = torch.load(os.path.join(self.args.cache_dir, split))
        else:
            if split == 'train':
                with jsonlines.open(os.path.join(self.args.dataset_dir, self.args.train_file), 'r') as jsonl_f:
                    dataset = [obj for obj in jsonl_f]
            elif split == 'val':
                with jsonlines.open(os.path.join(self.args.dataset_dir, self.args.val_file), 'r') as jsonl_f:
                    dataset = [obj for obj in jsonl_f]
            else:
                raise ValueError('no right dataset split')
            
            print('=====begin tokenize======')
            dataset = self.recursive_tokenize(dataset)
            print('=====end   tokenize======')
            torch.save(dataset, os.path.join(self.args.cache_dir, split))
        return dataset

    def collate_fn(self, batch):
        input_ids = []
        labels = []
        attention_mask = []
        bsz = len(batch)

        for instance in batch:
            batch_labels = []
            batch_input_ids = []
            token_facts = []
            tokens_question = [self.tokenizer.cls_token_id] + instance['Q'] + [self.tokenizer.sep_token_id]
            for pos_txt_fact in instance['pos_txt_facts']:
                token_facts.append(pos_txt_fact['fact'])
                batch_labels.append(1)
            for pos_img_fact in instance['pos_img_facts']:
                token_facts.append(pos_img_fact['caption'])
                batch_labels.append(1)
            neg_txt_count = 0
            neg_img_count = 0

            random.shuffle(instance['neg_txt_facts'])
            for neg_txt_fact in instance['neg_txt_facts']:
                if neg_txt_count < 8:
                    neg_txt_count += 1
                    token_facts.append(neg_txt_fact['fact'])
                    batch_labels.append(0)
            random.shuffle(instance['neg_img_facts'])
            for neg_img_fact in instance['neg_img_facts']:
                if neg_img_count < 8:
                    neg_img_count += 1
                    token_facts.append(neg_img_fact['caption'])
                    batch_labels.append(0)

            for token_fact in token_facts:
                batch_input_id = tokens_question + token_fact
                batch_input_id = batch_input_id[:self.args.max_length-1]
                batch_input_id += [self.tokenizer.sep_token_id]
                batch_input_id = torch.LongTensor(batch_input_id)
                batch_input_ids.append(batch_input_id)
            
            if len(batch_input_ids) > self.args.choice_num:
                batch_input_ids = batch_input_ids[:self.args.choice_num]
                batch_labels = batch_labels[:self.args.choice_num]
            else:
                num_placeholder = self.args.choice_num - len(batch_input_ids)
                batch_input_ids += [torch.LongTensor([self.tokenizer.pad_token_id]) for _ in range(num_placeholder)]
                batch_labels += [-100 for _ in range(num_placeholder)]

            input_ids += batch_input_ids # get (bsz x choice_num) x seq_len 
            labels += batch_labels

        input_ids = pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.LongTensor(labels)
        input_ids = input_ids.view(bsz, -1, input_ids.size(-1))
        labels = labels.view(bsz, -1)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        logit_mask = (labels != -100)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'logit_mask': logit_mask,
        }


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, SequentialSampler
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, default='./cache', help='the location of cache file')
    parser.add_argument('--have_cached_dataset', action='store_true')
    parser.add_argument('--dataset_dir', type=str, default='./data/')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='model name or path')
    parser.add_argument('--train_file', type=str, default='train.jsonl', help='path to train file, jsonl for scirex, conll for sciner')
    parser.add_argument('--val_file', type=str, default='val.jsonl', help='path to dev file')
    parser.add_argument('--test_file', type=str, default='val.jsonl', help='path to test file')
    parser.add_argument('--max_length', type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = WebQADataset(args, tokenizer, 'train')
    val_dataset = WebQADataset(args, tokenizer, 'val')
    train_dataloader = Dataloader(train_dataset,
                                  batch_size=2,
                                  sampler=SequentialSampler(val_dataset),
                                  collate_fn=train_dataset.collate_fn)
    val_dataloader = Dataloader(val_dataset,
                                batch_size=2,
                                sampler=SequentialSampler(val_dataset),
                                collate_fn=val_dataset.collate_fn)

    for data in train_dataloader:
        input_ids = data['input_ids']
        labels = data['labels']
        mask = data['attention_mask']
        # if input_ids.shape[1] != labels.shape[1] or labels.shape[1] != mask.shape[1]:
        print(input_ids)
        print(labels)
        print(mask)
        break
