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
            padding_value=self.tokenizer.pad_token_id
        )
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
            
            #print('=====begin tokenize======')
            #dataset = self.recursive_tokenize(dataset)
            #print('=====end   tokenize======')
            torch.save(dataset, os.path.join(self.args.cache_dir, split))
        return dataset

    def collate_fn(self, batch):
        sources = []
        targets = []
        prev_outputs = []
        labels = []
        constraint_masks = []
        bsz = len(batch)

        for instance in batch:
            batch_prev_outputs = []
            batch_targets = []
            batch_sources = []
            batch_labels = []
            batch_constraint_mask = []
            text_inputs = []
            text_outputs = []

            question = instance['Q']
            for pos_txt_fact in instance['pos_txt_facts']:
                text_inputs.append('Is text " {} " related to the question of " {} "?'.format(pos_txt_fact['fact'], question))
                text_outputs.append('yes')
                batch_labels.append(1)
            for pos_img_fact in instance['pos_img_facts']:
                text_inputs.append('Is image caption " {} " related to the question of " {} "?'.format(pos_img_fact['caption'], question))
                text_outputs.append('yes')
                batch_labels.append(1)

            neg_txt_count = 0
            neg_img_count = 0
            random.shuffle(instance['neg_txt_facts'])
            for neg_txt_fact in instance['neg_txt_facts']:
                if neg_txt_count < 8:
                    neg_txt_count += 1
                    text_inputs.append('Is text " {} " related to the question of " {} "?'.format(neg_txt_fact['fact'], question))
                    text_outputs.append('no')
                    batch_labels.append(0)

            random.shuffle(instance['neg_img_facts'])
            for neg_img_fact in instance['neg_img_facts']:
                if neg_img_count < 8:
                    neg_img_count += 1
                    text_inputs.append('Is image caption " {} " related to the question of " {} "?'.format(neg_img_fact['caption'], question))
                    text_outputs.append('no')
                    batch_labels.append(0)

            allowed_words = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(['yes', 'no']))
            for text_input, text_output in zip(text_inputs, text_outputs):
                source = self.tokenizer.encode(text_input, truncation=True, max_length=self.args.max_length, add_special_tokens=True)
                prev_output = source[:]
                target = self.tokenizer.encode(text_output, truncation=True, add_special_tokens=False)
                assert len(target) == 1
                target = prev_output[1:] + target
                constraint_mask = torch.zeros((len(prev_output), self.args.vocab_size)).bool()
                constraint_mask[-1][allowed_words] = True
                batch_sources.append(torch.LongTensor(source))
                batch_targets.append(torch.LongTensor(target))
                batch_prev_outputs.append(torch.LongTensor(prev_output))
                batch_constraint_mask.append(constraint_mask)

            if len(batch_sources) > self.args.choice_num:
                batch_sources = batch_sources[:self.args.choice_num]
                batch_targets = batch_targets[:self.args.choice_num]
                batch_prev_outputs = batch_prev_outputs[:self.args.choice_num]
                batch_constraint_mask = batch_constraint_mask[:self.args.choice_num]
                batch_labels = batch_labels[:self.args.choice_num]
            else:
                num_placeholder = self.args.choice_num - len(batch_sources)
                batch_sources += [torch.LongTensor([self.tokenizer.pad_token_id]) for _ in range(num_placeholder)]
                batch_targets += [torch.LongTensor([self.tokenizer.pad_token_id]) for _ in range(num_placeholder)]
                batch_prev_outputs += [torch.LongTensor([self.tokenizer.pad_token_id]) for _ in range(num_placeholder)]
                batch_constraint_mask += [torch.zeros((1, self.args.vocab_size)).bool() for _ in range(num_placeholder)]
                batch_labels += [-100 for _ in range(num_placeholder)]

            sources += batch_sources # get (bsz x choice_num) x seq_len 
            targets += batch_targets
            prev_outputs += batch_prev_outputs
            constraint_masks += batch_constraint_mask
            labels += batch_labels

        sources = pad_sequence(
            sources, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        targets = pad_sequence(
            targets,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        prev_outputs = pad_sequence(
            prev_outputs,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        constraint_masks = pad_sequence(
            constraint_masks,
            batch_first=True,
            padding_value=False,
        )

        sources = sources.view(bsz, -1, sources.size(-1))
        targets = targets.view(bsz, -1, targets.size(-1))
        prev_outputs = prev_outputs.view(bsz, -1, prev_outputs.size(-1))
        decoder_attention_mask = prev_outputs.ne(self.tokenizer.pad_token_id)
        labels = torch.LongTensor(labels)
        labels = labels.view(bsz, -1)
        logit_mask = (labels != -100)

        return {
            'sources': sources,
            'prev_outputs': prev_outputs,
            'targets': targets,
            'decoder_attention_mask': decoder_attention_mask,
            'constraint_masks': constraint_masks,
            'allowed_words': allowed_words,
            'labels': labels,
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
