import os
import jsonlines
import argparse
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


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
            elif split == 'test':
                with jsonlines.open(os.path.join(self.args.dataset_dir, self.args.test_file), 'r') as jsonl_f:
                    dataset = [obj for obj in jsonl_f]
            else:
                raise ValueError('no right dataset split')
            
            print('=====begin tokenize======')
            dataset = self.recursive_tokenize(dataset)
            print('=====end   tokenize======')
            torch.save(dataset, os.path.join(self.args.cache_dir, split))
        return dataset

    def collate_fn(self, batch, max_length=None):
        input_ids = []
        labels = []
        attention_mask = []

        for instance in batch:
            instance_token_ids = [self.tokenizer.cls_token_id]
            # QA part
            instance_token_ids += instance['Q']
            instance_token_ids += instance['A']
            # add one [SEP] after QA part
            instance_token_ids += [self.tokenizer.sep_token_id]
            # fact part (both for text and image)
            if 'txt_fact' in instance.keys():
                instance_token_ids += instance['txt_fact']['fact']
            elif 'img_fact' in instance.keys():
                instance_token_ids += instance['img_fact']['caption']
            else:
                raise ValueError('instance should either be image-based or text-based')

            instance_token_ids = instance_token_ids[:self.args.max_length]
            # add [SEP] after truncation
            instance_token_ids += [self.tokenizer.sep_token_id]
            instance_token_ids = torch.LongTensor(instance_token_ids)

            if 'txt_fact' in instance.keys():
                instance_labels = instance['txt_fact']['label']
            elif 'img_fact' in instance.keys():
                instance_labels = instance['img_fact']['label']
            else:
                raise ValueError('instance should either be image-based or text-based')

            input_ids.append(instance_token_ids)
            labels.append(instance_labels)

        input_ids = pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
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
