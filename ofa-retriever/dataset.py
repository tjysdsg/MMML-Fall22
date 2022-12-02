import os
import jsonlines
import argparse
import torch
import time
import torchvision
from torchvision import transforms
from PIL import Image, ImageFile
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class WebQATestDataset(Dataset):
    def __init__(
        self, 
        args, 
        tokenizer,
        imagenet_default_mean_and_std=False,
        patch_image_size=224,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.patch_image_size = patch_image_size

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert('RGB'),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.data = self.build_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index] 

    def build_dataset(self):
        if self.args.have_cached_dataset:
            dataset = torch.load(os.path.join(self.args.cache_dir, 'WebQA_test_dataset'))
        else:
            with jsonlines.open(os.path.join(self.args.dataset_dir, self.args.test_file), 'r') as jsonl_f:
                dataset = [obj for obj in jsonl_f]
            for data in tqdm(dataset):
                question = data['Q']
                if 'txt_fact' in data.keys():
                    text_input = 'Is text " {} " related to the question of " {} "?'.format(data['txt_fact']['fact'], question)
                    source = self.tokenizer.encode(text_input, truncation=True, max_length=self.args.max_length, add_special_tokens=True)
                    data['source'] = torch.LongTensor(source)
                    data['prev_output'] = torch.LongTensor(source)
                elif 'img_fact' in data.keys():
                    text_input = 'Is image caption " {} " related to the question of " {} "?'.format(data['img_fact']['caption'], question)
                    source = self.tokenizer.encode(text_input, truncation=True, max_length=self.args.max_length, add_special_tokens=True)
                    data['source'] = torch.LongTensor(source)
                    data['prev_output'] = torch.LongTensor(source)
            print('=' * 20 + 'Saving dataset' + '=' * 20)
            torch.save(dataset, os.path.join(self.args.cache_dir, 'WebQA_test_dataset'))
            print('=' * 20 + 'Saving dataset done' + '=' * 20)
        return dataset

    def collate_fn(self, batch, max_length=None):
        sources = []
        prev_outputs = []
        patch_images = []
        patch_masks = []
        q_ids = []
        source_types = []
        source_ids = []
        sources = []
        constraint_masks = []

        bsz = len(batch)

        allowed_words = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(['yes', 'no']))
        for instance in batch:
            q_ids.append(instance['Q_id'])
            if 'txt_fact' in instance.keys():
                source_types.append('txt')
                source_ids.append(instance['txt_fact']['snippet_id'])
                patch_image = torch.zeros((3, self.patch_image_size, self.patch_image_size))
                patch_mask = False
            elif 'img_fact' in instance.keys():
                source_types.append('img')
                source_ids.append(instance['img_fact']['image_id'])
                try:
                    image = Image.open(os.path.join(self.args.image_dir, str(instance['img_fact']['image_id']) + '.jpg'))
                    patch_image = self.patch_resize_transform(image)
                    patch_mask = True
                except:
                    patch_image = torch.zeros((3, self.patch_image_size, self.patch_image_size))
                    patch_mask = False
                    print('missing picture: {}, we need to ignore this.'.format(instance['img_fact']['image_id']))
            sources.append(instance['source'])
            prev_outputs.append(instance['prev_output'])
            patch_images.append(patch_image)
            patch_masks.append(patch_mask)

            constraint_mask = torch.zeros((len(instance['source']), self.args.vocab_size)).bool()
            constraint_mask[-1][allowed_words] = True
            constraint_masks.append(constraint_mask)

        sources = pad_sequence(
            sources, 
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

        patch_images = torch.stack(patch_images, dim=0)
        patch_masks = torch.BoolTensor(patch_masks)

        decoder_attention_mask = prev_outputs.ne(self.tokenizer.pad_token_id)
        return {
            'sources': sources,
            'prev_outputs': prev_outputs,
            'patch_masks': patch_masks,
            'patch_images': patch_images,
            'decoder_attention_mask': decoder_attention_mask,
            'source_ids': source_ids,
            'source_types': source_types,
            'q_ids': q_ids,
            'allowed_words': allowed_words,
            'constraint_masks': constraint_masks,
        }



class WebQADataset(Dataset):
    def __init__(
        self, 
        args, 
        tokenizer, 
        split, 
        imagenet_default_mean_and_std=False,
        patch_image_size=224,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.data = self.build_dataset(split=split)
        self.patch_image_size = patch_image_size

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert('RGB'),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

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
            
            torch.save(dataset, os.path.join(self.args.cache_dir, split))
        return dataset

    def collate_fn(self, batch):
        sources = []
        targets = []
        prev_outputs = []
        labels = []
        constraint_masks = []
        patch_images = []
        patch_masks = []
        bsz = len(batch)

        for instance in batch:
            batch_prev_outputs = []
            batch_targets = []
            batch_sources = []
            batch_labels = []
            batch_constraint_mask = []
            batch_patch_images = []
            batch_patch_masks = []
            text_inputs = []
            text_outputs = []

            question = instance['Q']
            for pos_txt_fact in instance['pos_txt_facts']:
                text_inputs.append('Is text " {} " related to the question of " {} "?'.format(pos_txt_fact['fact'], question))
                text_outputs.append('yes')
                batch_patch_images.append(torch.zeros((3, self.patch_image_size, self.patch_image_size)))
                batch_patch_masks.append(False)
                batch_labels.append(1)
            for pos_img_fact in instance['pos_img_facts']:
                text_inputs.append('Is image caption " {} " related to the question of " {} "?'.format(pos_img_fact['caption'], question))
                text_outputs.append('yes')
                image_id = pos_img_fact['image_id']
                image = Image.open(os.path.join(self.args.image_dir, str(image_id) + '.jpg'))
                batch_patch_images.append(self.patch_resize_transform(image))
                batch_patch_masks.append(True)
                batch_labels.append(1)

            neg_txt_count = 0
            neg_img_count = 0
            random.shuffle(instance['neg_txt_facts'])
            for neg_txt_fact in instance['neg_txt_facts']:
                if neg_txt_count < 8:
                    neg_txt_count += 1
                    text_inputs.append('Is text " {} " related to the question of " {} "?'.format(neg_txt_fact['fact'], question))
                    text_outputs.append('no')
                    batch_patch_images.append(torch.zeros((3, self.patch_image_size, self.patch_image_size)))
                    batch_patch_masks.append(False)
                    batch_labels.append(0)

            random.shuffle(instance['neg_img_facts'])
            for neg_img_fact in instance['neg_img_facts']:
                if neg_img_count < 8:
                    neg_img_count += 1
                    text_inputs.append('Is image caption " {} " related to the question of " {} "?'.format(neg_img_fact['caption'], question))
                    text_outputs.append('no')
                    image_id = neg_img_fact['image_id']
                    image = Image.open(os.path.join(self.args.image_dir, str(image_id) + '.jpg'))
                    batch_patch_images.append(image)
                    batch_patch_masks.append(True)
                    batch_labels.append(0)

            allowed_words = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(['yes', 'no']))
            for text_input, text_output in zip(text_inputs, text_outputs):
                source = self.tokenizer.encode(text_input, truncation=True, max_length=self.args.max_length, add_special_tokens=True)
                prev_output = source[:]
                target = self.tokenizer.encode(text_output, truncation=True, max_length=self.args.max_length, add_special_tokens=False)
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
                batch_patch_images = batch_patch_images[:self.args.choice_num]
                batch_patch_masks = batch_patch_masks[:self.args.choice_num] 
            else:
                num_placeholder = self.args.choice_num - len(batch_sources)
                batch_sources += [torch.LongTensor([self.tokenizer.pad_token_id]) for _ in range(num_placeholder)]
                batch_targets += [torch.LongTensor([self.tokenizer.pad_token_id]) for _ in range(num_placeholder)]
                batch_prev_outputs += [torch.LongTensor([self.tokenizer.pad_token_id]) for _ in range(num_placeholder)]
                batch_constraint_mask += [torch.zeros((1, self.args.vocab_size)).bool() for _ in range(num_placeholder)]
                batch_labels += [-100 for _ in range(num_placeholder)]
                batch_patch_images += [torch.zeros((3, self.patch_image_size, self.patch_image_size)) for _ in range(num_placeholder)]
                batch_patch_masks += [False for _ in range(num_placeholder)]

            sources += batch_sources # get (bsz x choice_num) x seq_len 
            targets += batch_targets
            prev_outputs += batch_prev_outputs
            constraint_masks += batch_constraint_mask
            labels += batch_labels
            patch_images += batch_patch_images
            patch_masks += batch_patch_masks

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
        patch_images = torch.stack(patch_images, dim=0)
        patch_images = patch_images.view(bsz, -1, patch_images.size(-3), patch_images.size(-2), patch_images.size(-1))
        patch_masks = torch.BoolTensor(patch_masks)
        patch_masks = patch_masks.view(bsz, -1)

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
            'patch_masks': patch_masks,
            'patch_images': patch_images,
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
