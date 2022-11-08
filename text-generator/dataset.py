import os
import jsonlines
import argparse
import torch
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class WebQATestDataset(Dataset):
    def __init__(self, 
                 args, 
                 tokenizer,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.args = args
        self.tokenizer = tokenizer
        self.data = self.build_dataset()
        self.question_prefix = tokenizer.encode(question_prefix, add_special_tokens=False)
        self.passage_prefix = tokenizer.encode(passage_prefix, add_special_tokens=False)

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
            with jsonlines.open(self.args.test_file, 'r') as jsonl_f:
                dataset = [obj for obj in jsonl_f]
            
            print('=====begin tokenize======')
            dataset = self.recursive_tokenize(dataset)
            print('=====end   tokenize======')
            torch.save(dataset, os.path.join(self.args.cache_dir, 'WebQA_test_dataset'))
        return dataset

    def collate_fn(self, batch, max_length=None):
        qids = []
        input_ids = []
        attention_mask = []
        bsz = len(batch)

        for instance in batch:
            token_facts = []
            tokens_question = self.question_prefix + instance['Q']
            for pos_txt_fact in instance['txt_facts']:
                token_facts += self.passage_prefix + pos_txt_fact['fact']
            for pos_img_fact in instance['img_facts']:
                token_facts += self.passage_prefix + pos_img_fact['caption']

            batch_input_ids = tokens_question + token_facts
            batch_input_ids = batch_input_ids[:self.args.encoder_max_length-1]
            batch_input_ids += [self.tokenizer.eos_token_id]
            batch_input_ids = torch.LongTensor(batch_input_ids)
            
            input_ids += [batch_input_ids] # get bsz x seq_len 

            qids += [instance['Q_id']]


        input_ids = pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            'qids': qids,
            'input_ids': input_ids.to(self.args.device),
            'attention_mask': attention_mask.to(self.args.device),
        }


class WebQADataset(Dataset):
    def __init__(self, 
                 args, 
                 tokenizer, 
                 split,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.args = args
        self.tokenizer = tokenizer
        self.data = self.build_dataset(split=split)
        self.question_prefix = tokenizer.encode(question_prefix, add_special_tokens=False)
        self.passage_prefix = tokenizer.encode(passage_prefix, add_special_tokens=False)

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
                if key == 'topic' or key == 'Qcate':
                    data[key] = value
                else:
                    data[key] = self.recursive_tokenize(value)
            return data
        
        lists = list(self.recursive_tokenize(subdata) for subdata in data)
        return lists    

    def build_dataset(self, split):
        if self.args.have_cached_dataset:
            dataset = torch.load(os.path.join(self.args.cache_dir, 'WebQA_{}_dataset'.format(split)))
        else:
            if split == 'train':
                with jsonlines.open(self.args.train_file, 'r') as jsonl_f:
                    dataset = [obj for obj in jsonl_f]
            elif split == 'val':
                with jsonlines.open(self.args.val_file, 'r') as jsonl_f:
                    dataset = [obj for obj in jsonl_f]
            else:
                raise ValueError('no right dataset split')
            
            print('=====begin tokenize======')
            dataset = self.recursive_tokenize(dataset)
            print('=====end   tokenize======')
            torch.save(dataset, os.path.join(self.args.cache_dir, 'WebQA_{}_dataset'.format(split)))
        return dataset

    def collate_fn(self, batch):
        Qcates = []
        input_ids = []
        labels = []
        attention_mask = []
        bsz = len(batch)

        for instance in batch:
            Qcate = instance['Qcate']
            Qcates.append(Qcate)

            token_facts = []
            tokens_question = self.question_prefix + instance['Q']
            for pos_txt_fact in instance['pos_txt_facts']:
                token_facts += self.passage_prefix + pos_txt_fact['fact']
            for pos_img_fact in instance['pos_img_facts']:
                token_facts += self.passage_prefix + pos_img_fact['caption']

            batch_input_ids = tokens_question + token_facts
            batch_input_ids = batch_input_ids[:self.args.encoder_max_length-1]
            batch_input_ids += [self.tokenizer.eos_token_id]
            batch_input_ids = torch.LongTensor(batch_input_ids)
            
            input_ids += [batch_input_ids] # get bsz x seq_len 

            tokens_answer = instance['A']
            batch_labels = tokens_answer  
            batch_labels = batch_labels[:self.args.decoder_max_length-1]
            batch_labels = torch.LongTensor(batch_labels)
            labels += [batch_labels] # get bsz x seq_len

        input_ids = pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )

        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )
        
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        decoder_attention_mask = labels.ne(-100)

        return {
            'input_ids': input_ids.to(self.args.device),
            'labels': labels.to(self.args.device),
            'attention_mask': attention_mask.to(self.args.device),
            'decoder_attention_mask': decoder_attention_mask.to(self.args.device),
            'Qcates': Qcates,
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
