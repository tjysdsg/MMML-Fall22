import random
import io
from random import randint, shuffle, choices
from random import random as rand
import pickle
import math
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from vlp.loader_utils import (
    get_random_word, batch_list_to_batch_tensors, Pipeline, TorchCPUUnpickler,
    get_image_feature_path
)
from typing import List
import os
import numpy as np


# FIXME: Refactor QA dataset, and processing code of QA samples


def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None,
                         always_truncate_tail=False):
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    while True:
        if len(tokens_a) <= max_len_a and len(tokens_b) <= max_len_b:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            del trunc_tokens[0]
            num_truncated[0] += 1
        else:
            trunc_tokens.pop()
            num_truncated[1] += 1
    return num_truncated_a, num_truncated_b


class webqaDataset_qa(Dataset):
    """ Load image feature path, q, a """

    def __init__(self, dataset_json_path, split, Qcate, batch_size, tokenizer, use_num_samples, processor, device=None):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.instance_list = []
        if device is not None:
            self.device = device
        assert os.path.exists(dataset_json_path), "loader.Dataset: dataset json file doesn't exist! {}".format(
            dataset_json_path)
        with open(dataset_json_path, "r") as f:
            dataset_J = json.load(f)
        count = 0
        for i in dataset_J:
            datum = dataset_J[i]
            if datum['split'] in split:  # modify here after we have split!!!!
                if not datum['Qcate'] == 'text': continue
                if ('all' in Qcate) or datum['Qcate'] in Qcate:
                    if use_num_samples == -1 or count < use_num_samples:
                        guid = datum['Guid']
                        qcate = datum['Qcate'] if 'Qcate' in datum else 'TBD'
                        Q = self.tokenizer.tokenize(datum['Q'].replace('"', ""))
                        A = self.tokenizer.tokenize(datum['A'][0].replace('"', ""))
                        A_list = [a.replace('"', "") for a in datum['A']]
                        try:
                            Keywords_A = datum['Keywords_A'].replace('"', "")
                        except:
                            Keywords_A = "TBD"
                        gold_facts = []
                        distractor_facts = []
                        for fa in datum['txt_posFacts']:
                            gold_facts.append(self.tokenizer.tokenize(fa['fact']))

                        self.instance_list.append((gold_facts, [], [], [], Q, A, Keywords_A, A_list, False, "txt", guid,
                                                   qcate))  # do_filter_task, context

                        count += 1

        print("Load {} instances from {} samples".format(len(self.instance_list), count))

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, idx):
        gold_facts, distractor_facts, gold_cxt_list, distractor_cxt_list, Q, A, Keywords_A, A_list, do_filter_task, context, example_id, Qcate = \
            self.instance_list[idx]
        instance = (
            gold_facts, distractor_facts, gold_cxt_list, distractor_cxt_list, Q, A, do_filter_task, context, example_id)
        instance = self.processor(instance, self.device)
        # Processor returns:
        # (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, 
        #       -1, is_distractor, self.task_idx, img, vis_pe, context_is_img)
        return instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.instance_list) / float(self.batch_size))):
            batch = []
            for _ in range(self.batch_size):
                idx = randint(0, len(self.instance_list) - 1)  # allow overlap between batches???
                batch.append(self.__getitem__(idx))
            yield batch_list_to_batch_tensors(batch)

    def get_QA_list(self):
        return [i[4] for i in self.instance_list], [i[7] for i in self.instance_list], [i[6] for i in
                                                                                        self.instance_list]

    def get_guid_list(self):
        return [i[-2] for i in self.instance_list]

    def get_Qcate_list(self):
        return [i[-1] for i in self.instance_list]


class webqaDataset_qa_with_img(Dataset):
    def __init__(self, dataset_json_path, split, Qcate, batch_size, tokenizer, feature_folder, use_num_samples,
                 processor, imgid_map=None, device=None):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.instance_list = []
        if imgid_map is not None:
            self.imgid_map = pickle.load(open(imgid_map, "rb"))
            print("\nLoad imgid_map, length={}\n".format(len(self.imgid_map)))
        else:
            self.imgid_map = None
        if device is not None:
            self.device = device
        assert os.path.exists(dataset_json_path), "loader.Dataset: dataset json file doesn't exist!"
        assert os.path.exists(feature_folder), "loader.Dataset: feature folder doesn't exist!"

        with open(dataset_json_path, "r") as f:
            dataset_J = json.load(f)
        count = 0
        for i in dataset_J:
            datum = dataset_J[i]
            if datum['split'] in split:
                if datum['Qcate'] == 'text': continue
                if ('all' in Qcate) or datum['Qcate'] in Qcate:
                    if use_num_samples == -1 or count < use_num_samples:
                        guid = datum['Guid']
                        qcate = datum['Qcate'] if 'Qcate' in datum else 'TBD'
                        Q = self.tokenizer.tokenize(datum['Q'].replace('"', ""))
                        A = self.tokenizer.tokenize(datum['A'][0].replace('"', ""))
                        A_list = [a.replace('"', "") for a in datum['A']]
                        try:
                            Keywords_A = datum['Keywords_A'].replace('"', "")
                        except:
                            Keywords_A = "TBD"

                        gold_feature_paths = []
                        gold_cxt_list = []
                        for im in datum['img_posFacts']:
                            image_id = int(im['image_id'])
                            if self.imgid_map is not None:
                                image_id = self.imgid_map[image_id]
                            image_feature_path = get_image_feature_path(feature_folder, image_id)

                            gold_feature_paths.append(image_feature_path)
                            cxt = self.tokenizer.tokenize(im['caption'].strip())
                            gold_cxt_list.append(cxt)
                        self.instance_list.append((gold_feature_paths, [], gold_cxt_list, [], Q, A, Keywords_A, A_list,
                                                   False, "img", guid, qcate))  # do_filter_task, context )
                        count += 1

        print("Load {} instances from {} samples".format(len(self.instance_list), count))

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, idx):
        gold_facts, distractor_facts, gold_cxt_list, distractor_cxt_list, Q, A, Keywords_A, A_list, do_filter_task, context, example_id, Qcate = \
            self.instance_list[idx]
        instance = (
            gold_facts, distractor_facts, gold_cxt_list, distractor_cxt_list, Q, A, do_filter_task, context, example_id)
        instance = self.processor(instance, self.device)
        # Processor returns:
        # (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, 
        #       -1, is_distractor, self.task_idx, img, vis_pe, context_is_img)
        return instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.instance_list) / float(self.batch_size))):
            batch = []
            for _ in range(self.batch_size):
                idx = randint(0, len(self.instance_list) - 1)  # allow overlap between batches???
                batch.append(self.__getitem__(idx))
            yield batch_list_to_batch_tensors(batch)

    def get_QA_list(self):
        return [i[4] for i in self.instance_list], [i[7] for i in self.instance_list], [i[6] for i in
                                                                                        self.instance_list]

    def get_guid_list(self):
        return [i[-2] for i in self.instance_list]

    def get_Qcate_list(self):
        return [i[-1] for i in self.instance_list]


class WebQADataSampleProcessor(Pipeline):
    def __init__(self, max_pred, mask_prob, vocab_words, indexer, seed, max_len, len_vis_input, max_len_a, max_len_b,
                 max_len_img_cxt=200, new_segment_ids=True, truncate_config=None, use_img_meta=True,
                 use_img_content=True, use_txt_fact=True):
        super().__init__()
        if truncate_config is None:
            truncate_config = {}

        self.task_idx = 3  # use task_idx for s2s in relaxed projection layer
        self.max_pred = max_pred
        self.mask_prob = mask_prob
        self.len_vis_input = len_vis_input
        self.vocab_words = vocab_words
        self.indexer = indexer
        self.max_len_img_cxt = max_len_img_cxt
        self._tril_matrix = torch.tril(torch.ones((max_len, max_len), dtype=torch.long))
        self.always_truncate_tail = truncate_config.get('always_truncate_tail', False)
        self.max_len_b = max_len_b
        self.max_len_a = max_len_a
        self.max_len = max_len
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.new_segment_ids = new_segment_ids
        self.use_img_meta = use_img_meta
        self.use_img_content = use_img_content
        self.use_txt_fact = use_txt_fact
        random.seed(seed)
        np.random.seed(seed)
        assert max_len_a + max_len_b <= max_len, "loader Processor: max_len_a + max_len_b > max_len"

    def detokenize(self, tk_list):
        r_list = []
        for tk in tk_list:
            if tk.startswith('##') and len(r_list) > 0:
                r_list[-1] = r_list[-1] + tk[2:]
            else:
                r_list.append(tk)
        return r_list

    def load_image_feats(self, path: str, pad=True):
        """
        Load image features, normalize coordinates, and optionally pad to max length
        """
        with open(path, "rb") as f:
            features = TorchCPUUnpickler(f).load()

        img = features['fc1_features'].detach().cpu().float()
        cls_label = features['cls_features'].detach().cpu().float()
        vis_pe = features['pred_boxes'].detach().cpu()

        # Lazy normalization of the coordinates
        w_est = torch.max(vis_pe[:, [0, 2]]) * 1. + 1e-5
        h_est = torch.max(vis_pe[:, [1, 3]]) * 1. + 1e-5
        # assert h_est > 0, f'box h_est should greater than 0! {h_est}'
        # assert w_est > 0, f'box w_est should greater than 0! {w_est}'
        vis_pe[:, [0, 2]] /= w_est
        vis_pe[:, [1, 3]] /= h_est
        rel_area = (vis_pe[:, 3] - vis_pe[:, 1]) * (vis_pe[:, 2] - vis_pe[:, 0])
        rel_area.clamp_(0)

        vis_pe = torch.cat(
            (
                vis_pe[:, :4], rel_area.view(-1, 1), features['scores'].detach().cpu().view(-1, 1)
            ),
            -1
        )
        vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), F.layer_norm(cls_label, [1601])), dim=-1)

        assert img.size(0) == vis_pe.size(0), "img features and vis_pe should have the same token length!"

        if pad:
            vis_pad = torch.zeros((self.max_len_img_cxt - img.size(0), img.size(-1)))
            img = torch.cat((img, vis_pad), dim=0)
            pe_pad = torch.zeros((self.max_len_img_cxt - vis_pe.size(0), vis_pe.size(-1)))
            vis_pe = torch.cat((vis_pe, pe_pad), dim=0)

            # assert vis_pe.size(0) == self.max_len_img_cxt
            # assert img.size(0) == self.max_len_img_cxt

        return img, vis_pe

    def load_filter_data(self, instance, filter_max_choices=None) -> dict:
        _, _, _, _, _, _, _, modality_type, _ = instance
        assert filter_max_choices is not None, "must pass in a valid filter_max_choices when doing filter task"
        if modality_type == 'both':
            (
                gold_text_facts, distractor_text_facts, gold_img_and_caps, distractor_img_and_caps,
                Q, A, do_filter_task, _, example_id,
            ) = instance

            ori_choices = []  # unique identifiers of chosen facts
            input_ids_list = []  # token sequence that will be the input of Bert
            input_mask_list = []  # self attention masks
            segment_ids_list = []
            img_list = []  # list of image feature
            vis_pe_list = []  # list of image positional embeddings

            # permuted order of the pos/neg text/img fact sequence
            # pick one type of fact at a time following this order
            # 0: positive txt
            # 1: positive img
            # 2: negative txt
            # 3: negative img
            order = [0] * len(gold_text_facts) \
                    + [1] * len(gold_img_and_caps) \
                    + [2] * len(distractor_text_facts) \
                    + [3] * len(distractor_img_and_caps)
            order = np.random.permutation(order)

            # 1 = positive fact, 0 negative fact
            label = torch.tensor([1. if o <= 1 else 0. for o in order])
            label = torch.stack([label, 1 - label], dim=0).transpose(1, 0)

            for o in order:

                if o == 1 or o == 3:  # context is img
                    if o == 1:  # pos img
                        img_path, caption = gold_img_and_caps.pop()
                    else:  # neg img
                        img_path, caption = distractor_img_and_caps.pop()

                    ori_choices.append(img_path.split('/')[-1].replace('.pkl', ''))

                    tokens_a = ['[UNK]'] * self.max_len_img_cxt
                    tokens_b = Q + A

                    # truncate
                    max_len_cxt_meta = self.max_len_a - self.max_len_img_cxt
                    truncate_tokens_pair(caption, tokens_b, max_len=max_len_cxt_meta + self.max_len_b,
                                         max_len_a=max_len_cxt_meta, max_len_b=self.max_len_b,
                                         trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)
                    if self.use_img_meta:
                        tokens_a += caption

                    tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

                    if self.new_segment_ids:
                        segment_ids = [4] * (len(tokens_a) + 2) + [5] * (len(tokens_b) + 1)
                    else:
                        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

                    # self-attention mask
                    input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
                    # Any token can attend to img, img caption and question
                    # Nothing attends to A for filter task
                    img_end_pos = 1 + self.len_vis_input
                    if self.use_img_content:
                        input_mask[:, :img_end_pos].fill_(1)
                    st, end = 1 + self.max_len_img_cxt, len(tokens_a) + 2 + len(Q)
                    input_mask[:, st:end].fill_(1)
                    input_mask_list.append(input_mask)

                    input_ids = self.indexer(tokens)
                    n_pad = self.max_len - len(input_ids)
                    input_ids.extend([0] * n_pad)
                    segment_ids.extend([0] * n_pad)
                    input_ids_list.append(torch.tensor(input_ids))
                    segment_ids_list.append(torch.tensor(segment_ids))

                    img, vis_pe = self.load_image_feats(img_path)
                    if not self.use_img_content:
                        img = torch.zeros_like(img).float()
                        vis_pe = torch.zeros_like(vis_pe).float()
                    img_list.append(img)
                    vis_pe_list.append(vis_pe)

                else:  # text
                    tokens_a = []
                    if self.use_txt_fact:
                        if o == 0:  # pos text
                            f = gold_text_facts.pop()
                            tokens_a = f['fact']
                            ori_choices.append(f['snippet_id'])
                        else:  # neg text
                            f = distractor_text_facts.pop()
                            tokens_a = f['fact']
                            ori_choices.append(f['snippet_id'])

                    tokens_b = Q + A
                    # truncate
                    truncate_tokens_pair(tokens_a, tokens_b, max_len=self.max_len_a + self.max_len_b,
                                         max_len_a=self.max_len_a, max_len_b=self.max_len_b,
                                         trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)

                    tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

                    if self.new_segment_ids:
                        segment_ids = [4] * (len(tokens_a) + 2) + [5] * (len(tokens_b) + 1)
                    else:
                        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

                    # self-attention mask
                    input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
                    # Any token can attend to text and question
                    # Nothing attends to A for filter task
                    input_mask[:, :len(tokens_a) + 2 + len(Q)].fill_(1)
                    input_mask_list.append(input_mask)

                    input_ids = self.indexer(tokens)
                    n_pad = self.max_len - len(input_ids)
                    input_ids.extend([0] * n_pad)
                    segment_ids.extend([0] * n_pad)

                    input_ids_list.append(torch.tensor(input_ids))
                    segment_ids_list.append(torch.tensor(segment_ids))

            # pad facts to filter_max_choices length
            logit_mask = [1.] * len(input_ids_list)
            if len(input_ids_list) < filter_max_choices:
                num_placeholder = filter_max_choices - len(input_ids_list)
                input_ids_list.extend([input_ids_list[-1]] * num_placeholder)
                segment_ids_list.extend([segment_ids_list[-1]] * num_placeholder)
                input_mask_list.extend([input_mask_list[-1]] * num_placeholder)
                logit_mask.extend([0.] * num_placeholder)
                label = torch.cat([label, torch.tensor([[0., 0.]] * num_placeholder)], dim=0)

            input_ids = torch.stack(input_ids_list, dim=0)
            segment_ids = torch.stack(segment_ids_list, dim=0)
            input_mask = torch.stack(input_mask_list, dim=0)
            logit_mask = torch.tensor(logit_mask)

            if len(img_list) == 0:
                img = None
                vis_pe = None
            else:
                img = torch.stack(img_list, dim=0)
                vis_pe = torch.stack(vis_pe_list, dim=0)

            # Sequence indices where image facts are located
            cxt_modality_label = [i for i in range(len(order)) if order[i] % 2 == 1]
            assert len(cxt_modality_label)

            return {"input_ids": input_ids,
                    "segment_ids": segment_ids,
                    "input_mask": input_mask,
                    "masked_ids": None, "masked_pos": None, "masked_weights": None,
                    "is_next_label": -1,
                    "do_filter_task": do_filter_task,
                    "filter_label": label,
                    "logit_mask": logit_mask,
                    "ori_choices": ori_choices,
                    "task_idx": self.task_idx,
                    "img": img,
                    "vis_pe": vis_pe,
                    "context": modality_type,
                    "cxt_modality_label": cxt_modality_label,
                    "example_id": example_id}

        elif modality_type == 'img':
            (
                _, _, gold_image_facts, distractor_image_facts,
                Q, A, do_filter_task, _, example_id,
            ) = instance

            input_ids_list = []
            segment_ids_list = []
            input_mask_list = []
            img_list = []
            vis_pe_list = []

            num_gold = len(gold_image_facts)
            filter_num_choices = num_gold + len(distractor_image_facts)
            perm = np.random.permutation(filter_num_choices)
            all_image_choices = gold_image_facts + distractor_image_facts
            all_image_choices = [all_image_choices[p] for p in perm]

            label = torch.tensor([1. if p < num_gold else 0. for p in perm])
            label = torch.stack([label, 1 - label], dim=0).transpose(1, 0)

            for i in range(filter_num_choices):
                img_path, captions = all_image_choices[i]

                tokens_a = ['[UNK]'] * self.max_len_img_cxt
                tokens_b = Q + A

                # truncate
                max_len_cxt_meta = self.max_len_a - self.max_len_img_cxt
                truncate_tokens_pair(captions, tokens_b, max_len=max_len_cxt_meta + self.max_len_b,
                                     max_len_a=max_len_cxt_meta, max_len_b=self.max_len_b, trunc_seg=self.trunc_seg,
                                     always_truncate_tail=self.always_truncate_tail)
                if self.use_img_meta:
                    tokens_a += captions

                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

                if self.new_segment_ids:
                    segment_ids = [4] * (len(tokens_a) + 2) + [5] * (len(tokens_b) + 1)
                else:
                    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

                # self-attention mask
                input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
                # everyone can attend to img, cxt_meta and Q. Nobody cares attention to A for filter task
                img_end_pos = 1 + self.len_vis_input
                if self.use_img_content: input_mask[:, :img_end_pos].fill_(1)
                st, end = 1 + self.max_len_img_cxt, len(tokens_a) + 2 + len(Q)
                input_mask[:, st:end].fill_(1)
                input_ids = self.indexer(tokens)
                n_pad = self.max_len - len(input_ids)
                input_ids.extend([0] * n_pad)
                segment_ids.extend([0] * n_pad)

                img, vis_pe = self.load_image_feats(img_path)

                input_ids_list.append(torch.tensor(input_ids))
                segment_ids_list.append(torch.tensor(segment_ids))
                input_mask_list.append(input_mask)
                if not self.use_img_content:
                    img = torch.zeros_like(img).float()
                    vis_pe = torch.zeros_like(vis_pe).float()
                img_list.append(img)
                vis_pe_list.append(vis_pe)

            logit_mask = [1.] * len(input_ids_list)
            if len(input_ids_list) < filter_max_choices:
                num_placeholder = filter_max_choices - len(input_ids_list)
                input_ids_list.extend([input_ids_list[-1]] * num_placeholder)
                segment_ids_list.extend([segment_ids_list[-1]] * num_placeholder)
                input_mask_list.extend([input_mask_list[-1]] * num_placeholder)
                logit_mask.extend([0.] * num_placeholder)
                label = torch.cat([label, torch.tensor([[0., 0.]] * num_placeholder)], dim=0)

            input_ids = torch.stack(input_ids_list, dim=0)
            segment_ids = torch.stack(segment_ids_list, dim=0)
            input_mask = torch.stack(input_mask_list, dim=0)
            img = torch.stack(img_list, dim=0)
            vis_pe = torch.stack(vis_pe_list, dim=0)
            logit_mask = torch.tensor(logit_mask)

            ori_choices = [i.split('/')[-1].replace('.pkl', '') for i in all_image_choices]

            cxt_modality_label = range(filter_num_choices)
            return {"input_ids": input_ids,
                    "segment_ids": segment_ids,
                    "input_mask": input_mask,
                    "masked_ids": None, "masked_pos": None, "masked_weights": None,
                    "is_next_label": -1,
                    "do_filter_task": do_filter_task,
                    "filter_label": label,
                    "logit_mask": logit_mask,
                    "ori_choices": ori_choices,
                    "task_idx": self.task_idx,
                    "img": img,
                    "vis_pe": vis_pe,
                    "context": modality_type,
                    "cxt_modality_label": cxt_modality_label,
                    "example_id": example_id}

        elif modality_type == 'txt':
            gold_text_facts, distractor_text_facts, _, _, Q, A, do_filter_task, _, example_id = instance

            num_gold = len(gold_text_facts)
            filter_num_choices = num_gold + len(distractor_text_facts)
            perm = np.random.permutation(filter_num_choices)
            all_choices_facts = [f['fact'] for f in gold_text_facts + distractor_text_facts]
            all_choices_facts = [all_choices_facts[p] for p in perm]
            ori_choices = [f['snippet_id'] for f in gold_text_facts + distractor_text_facts]
            ori_choices = [ori_choices[p] for p in perm]

            label = torch.tensor([1. if p < num_gold else 0. for p in perm])
            label = torch.stack([label, 1 - label], dim=0).transpose(1, 0)
            input_ids_list = []
            segment_ids_list = []
            input_mask_list = []
            for i in range(filter_num_choices):
                tokens_a = []
                if self.use_txt_fact:
                    tokens_a = all_choices_facts[i].copy()
                tokens_b = Q + A

                # truncate
                truncate_tokens_pair(tokens_a, tokens_b, max_len=self.max_len_a + self.max_len_b,
                                     max_len_a=self.max_len_a, max_len_b=self.max_len_b, trunc_seg=self.trunc_seg,
                                     always_truncate_tail=self.always_truncate_tail)

                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

                if self.new_segment_ids:
                    segment_ids = [4] * (len(tokens_a) + 2) + [5] * (len(tokens_b) + 1)
                else:
                    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

                # self-attention mask
                input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
                # everyone can attend to cxt and Q. Nobody cares attention to A for filter task
                input_mask[:, :len(tokens_a) + 2 + len(Q)].fill_(1)

                input_ids = self.indexer(tokens)
                n_pad = self.max_len - len(input_ids)
                input_ids.extend([0] * n_pad)
                segment_ids.extend([0] * n_pad)

                input_ids_list.append(torch.tensor(input_ids))
                segment_ids_list.append(torch.tensor(segment_ids))
                input_mask_list.append(input_mask)

            logit_mask = [1.] * len(input_ids_list)
            if len(input_ids_list) < filter_max_choices:
                num_placeholder = filter_max_choices - len(input_ids_list)
                input_ids_list.extend([input_ids_list[-1]] * num_placeholder)
                segment_ids_list.extend([segment_ids_list[-1]] * num_placeholder)
                input_mask_list.extend([input_mask_list[-1]] * num_placeholder)
                logit_mask.extend([0.] * num_placeholder)
                label = torch.cat([label, torch.tensor([[0., 0.]] * num_placeholder)], dim=0)
            input_ids = torch.stack(input_ids_list, dim=0)
            segment_ids = torch.stack(segment_ids_list, dim=0)
            input_mask = torch.stack(input_mask_list, dim=0)
            logit_mask = torch.tensor(logit_mask)

            return {"input_ids": input_ids,
                    "segment_ids": segment_ids,
                    "input_mask": input_mask,
                    "masked_ids": None, "masked_pos": None, "masked_weights": None,
                    "is_next_label": -1,
                    "do_filter_task": do_filter_task,
                    "filter_label": label,
                    "logit_mask": logit_mask,
                    "ori_choices": ori_choices,
                    "task_idx": self.task_idx,
                    "img": None,
                    "vis_pe": None,
                    "context": modality_type,
                    "cxt_modality_label": [],
                    "example_id": example_id}

    def __call__(self, instance, filter_max_choices=None, device=None):
        _, _, _, _, _, _, do_filter_task, context, example_id = instance

        # Retrieval
        if do_filter_task:
            return self.load_filter_data(instance, filter_max_choices)

        # QA
        else:
            if context == 'img':
                gold_feature_paths, distractor_feature_paths, gold_cxt_list, distractor_cxt_list, Q, A, do_filter_task, context, example_id = instance
                gold_feature_paths = gold_feature_paths[:2]
                gold_cxt_list = gold_cxt_list[:2]
                tokens_a = ['[UNK]'] * self.max_len_img_cxt
                tokens_b = Q + A

                cxt = sum(gold_cxt_list, [])

                num_truncated_a, num_truncated_b = truncate_tokens_pair(cxt, tokens_b,
                                                                        max_len=self.max_len_a - self.max_len_img_cxt + self.max_len_b,
                                                                        max_len_a=self.max_len_a - self.max_len_img_cxt,
                                                                        max_len_b=self.max_len_b,
                                                                        trunc_seg=self.trunc_seg,
                                                                        always_truncate_tail=self.always_truncate_tail)
                if self.use_img_meta: tokens_a += cxt
                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
                if self.new_segment_ids:
                    segment_ids = [4] * (len(tokens_a) + 2) + [5] * (len(tokens_b) + 1)
                else:
                    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

                effective_len_A = len(A) + 1 - num_truncated_b[1]
                n_pred = min(self.max_pred, max(min(3, effective_len_A), int(round(
                    effective_len_A * self.mask_prob))))  # predict everything if answer has less than 3 tokens
                cand_pos = []
                for i, tk in enumerate(tokens):
                    # only mask tk in A
                    if (i >= len(tokens_a) + 2 + len(Q) - num_truncated_b[0]):
                        cand_pos.append(i)

                shuffle(cand_pos)
                masked_pos = cand_pos[:n_pred]
                masked_tokens = [tokens[pos] for pos in masked_pos]  # gth token in masked_pos
                for pos in masked_pos:
                    if rand() < 0.8:
                        tokens[pos] = '[MASK]'
                    elif rand() < 0.5:
                        random_word = get_random_word(self.vocab_words)
                        tokens[pos] = random_word

                masked_weights = [1] * len(masked_tokens)
                masked_ids = self.indexer(masked_tokens)

                input_ids = self.indexer(tokens)
                n_pad = self.max_len - len(input_ids)
                input_ids.extend([0] * n_pad)
                segment_ids.extend([0] * n_pad)

                # self-attention mask
                num_img = len(gold_feature_paths)
                input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)

                img_end_pos = 1 + self.len_vis_input * num_img
                if self.use_img_content: input_mask[:, :img_end_pos].fill_(1)
                st, end = 1 + self.max_len_img_cxt, 2 + len(tokens_a) + len(Q)
                input_mask[:, st:end].fill_(1)
                # Tokens in A can attend to previous tokens in A
                pred_st, pred_end = 2 + len(tokens_a) + len(Q), len(tokens)
                input_mask[pred_st:pred_end, pred_st:pred_end].copy_(
                    self._tril_matrix[:pred_end - pred_st, :pred_end - pred_st])
                # Zero padding for masked target

                if self.max_pred > n_pred:
                    n_pad = self.max_pred - n_pred
                    masked_ids.extend([0] * n_pad)
                    masked_pos.extend([0] * n_pad)
                    masked_weights.extend([0] * n_pad)

                # Convert some inputs to tensors
                input_ids = torch.LongTensor(input_ids)
                segment_ids = torch.LongTensor(segment_ids)
                masked_ids = torch.LongTensor(masked_ids)
                masked_pos = torch.LongTensor(masked_pos)
                masked_weights = torch.LongTensor(masked_weights)

                img_list = []
                vis_pe_list = []
                for img_path in gold_feature_paths:
                    assert os.path.exists(img_path), "loader Processor: .pkl file doesn't exist! {}".format(img_path)

                    img, vis_pe = self.load_image_feats(img_path)
                    img_list.append(img)
                    vis_pe_list.append(vis_pe)
                img = torch.cat(img_list, dim=0)
                vis_pe = torch.cat(vis_pe_list, dim=0)

                if len(masked_pos) < self.max_pred:
                    print("num_truncated_b = ", num_truncated_b)
                    print(masked_pos)
                    print(n_pred)
                    print(self.max_pred)
                    print("effective_len_A = ", effective_len_A)
                    print("len(A) = ", len(A))
                    print("len(Q) = ", len(Q))
                    print("--------------")
                    print(tokens)
                cxt_modality_label = [1]
                # schema: (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next_label, do_filter_task, filter_label, logit_mask, ori_choices, self.task_idx, img, vis_pe, context, cxt_modality_label, example_id)
                return {"input_ids": input_ids,
                        "segment_ids": segment_ids,
                        "input_mask": input_mask,
                        "masked_ids": masked_ids, "masked_pos": masked_pos, "masked_weights": masked_weights,
                        "is_next_label": -1,
                        "do_filter_task": do_filter_task,
                        "filter_label": None,
                        "logit_mask": None,
                        "ori_choices": None,
                        "task_idx": self.task_idx,
                        "img": img,
                        "vis_pe": vis_pe,
                        "context": context,
                        "cxt_modality_label": cxt_modality_label,
                        "example_id": example_id}

            else:  # qa task, context is txt
                gold_text_facts, distractor_text_facts, gold_cxt_list, distractor_cxt_list, Q, A, do_filter_task, context, example_id = instance
                tokens_a = []
                if self.use_txt_fact: tokens_a = sum(gold_text_facts, [])
                tokens_b = Q + A
                num_truncated_a, num_truncated_b = truncate_tokens_pair(tokens_a, tokens_b,
                                                                        max_len=self.max_len_a + self.max_len_b,
                                                                        max_len_a=self.max_len_a,
                                                                        max_len_b=self.max_len_b,
                                                                        trunc_seg=self.trunc_seg,
                                                                        always_truncate_tail=self.always_truncate_tail)
                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
                if self.new_segment_ids:
                    segment_ids = [4] * (len(tokens_a) + 2) + [5] * (len(tokens_b) + 1)
                else:
                    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

                effective_len_A = len(A) + 1 - num_truncated_b[1]
                n_pred = min(self.max_pred, max(1, int(round(effective_len_A * self.mask_prob))))
                cand_pos = []
                for i, tk in enumerate(tokens):
                    # only mask tk in A
                    if (i >= len(tokens_a) + 2 + len(Q) - num_truncated_b[0]):
                        cand_pos.append(i)

                shuffle(cand_pos)
                masked_pos = cand_pos[:n_pred]
                masked_tokens = [tokens[pos] for pos in masked_pos]  # gth token in masked_pos
                for pos in masked_pos:
                    if rand() < 0.8:
                        tokens[pos] = '[MASK]'
                    elif rand() < 0.5:
                        tokens[pos] = get_random_word(self.vocab_words)

                masked_weights = [1] * len(masked_tokens)
                masked_ids = self.indexer(masked_tokens)

                input_ids = self.indexer(tokens)
                n_pad = self.max_len - len(input_ids)
                input_ids.extend([0] * n_pad)
                segment_ids.extend([0] * n_pad)

                input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
                input_mask[:, :len(tokens_a) + 2 + len(Q)].fill_(1)
                pred_st, pred_end = 2 + len(tokens_a) + len(Q), len(tokens)
                input_mask[pred_st:pred_end, pred_st:pred_end].copy_(
                    self._tril_matrix[:pred_end - pred_st, :pred_end - pred_st])

                # Zero padding for masked target
                if self.max_pred > n_pred:
                    n_pad = self.max_pred - n_pred
                    masked_ids.extend([0] * n_pad)
                    masked_pos.extend([0] * n_pad)
                    masked_weights.extend([0] * n_pad)

                # Convert some inputs to tensors
                input_ids = torch.LongTensor(input_ids)
                segment_ids = torch.LongTensor(segment_ids)
                masked_ids = torch.LongTensor(masked_ids)
                masked_pos = torch.LongTensor(masked_pos)
                masked_weights = torch.LongTensor(masked_weights)

                # schema: (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next_label, do_filter_task, filter_label, logit_mask, ori_choices, self.task_idx, img, vis_pe, context, cxt_modality_label, example_id)
                return {"input_ids": input_ids,
                        "segment_ids": segment_ids,
                        "input_mask": input_mask,
                        "masked_ids": masked_ids, "masked_pos": masked_pos, "masked_weights": masked_weights,
                        "is_next_label": -1,
                        "do_filter_task": do_filter_task,
                        "filter_label": None,
                        "logit_mask": None,
                        "ori_choices": None,
                        "task_idx": self.task_idx,
                        "img": None,
                        "vis_pe": None,
                        "context": context,
                        "cxt_modality_label": None,
                        "example_id": example_id}


class Preprocess4webqaDecoder(Pipeline):

    def __init__(self, vocab_words, indexer, seed, max_len, len_vis_input, max_len_a, max_len_Q, max_len_img_cxt=200,
                 max_tgt_len=30, new_segment_ids=True, truncate_config={}, use_img_meta=True, use_img_content=True,
                 use_txt_fact=True):
        super().__init__()
        self.task_idx = 3  # use task_idx for s2s in relaxed projection layer
        self.len_vis_input = len_vis_input
        self.vocab_words = vocab_words
        self.indexer = indexer
        self.max_len_img_cxt = max_len_img_cxt
        self._tril_matrix = torch.tril(torch.ones((max_len, max_len), dtype=torch.long))
        self.always_truncate_tail = truncate_config.get('always_truncate_tail', False)
        self.max_len_Q = max_len_Q
        self.max_len_a = max_len_a
        self.max_len = min(max_len, max_len_a + 2 + max_len_Q + max_tgt_len)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        assert max_len_a + max_len_Q <= max_len, "loader Processor: max_len_a + max_len_b > max_len"
        self.new_segment_ids = new_segment_ids
        self.use_img_meta = use_img_meta
        self.use_img_content = use_img_content
        self.use_txt_fact = use_txt_fact
        random.seed(seed)
        np.random.seed(seed)

    def __call__(self, instance, filter_max_choices=None, device=None):
        _, __, ___, ____, _____, ______, do_filter_task, context, example_id = instance
        if do_filter_task:
            raise ValueError(
                "Processor for decoder does not support filter task. \nFor filter task inference, please use run_webqa.py by setting args.do_train=False")
        else:
            if context in ['img', 'both']:
                gold_feature_paths, distractor_feature_paths, gold_cxt_list, distractor_cxt_list, Q, A, do_filter_task, context, example_id = instance
                gold_feature_paths = gold_feature_paths[:2]
                gold_cxt_list = gold_cxt_list[:2]
                tokens_a = ['[UNK]'] * self.max_len_img_cxt
                cxt = sum(gold_cxt_list, [])

                tokens_b = Q.copy()  # without copy Q will change as we modify tokens_b during padding!!!!!
                truncate_tokens_pair(cxt, tokens_b, max_len=self.max_len_a - self.max_len_img_cxt + self.max_len_Q,
                                     max_len_a=self.max_len_a - self.max_len_img_cxt, max_len_b=self.max_len_Q,
                                     trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)
                if self.use_img_meta: tokens_a += cxt

                n_pad = self.max_len_Q + self.max_len_a - len(tokens_a) - len(tokens_b)
                tokens_b += ['[PAD]'] * n_pad

                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b  # + ['[SEP]'] # start generating right after Q
                if self.new_segment_ids:
                    segment_ids = [4] * (len(tokens_a) + 2) + [5] * len(tokens_b) + [5] * (self.max_len - len(tokens))
                else:
                    segment_ids = [0] * (len(tokens_a) + 2) + [1] * len(tokens_b) + [5] * (self.max_len - len(tokens))

                # position_ids
                ori_Q_len = min(len(Q), self.max_len_Q)
                position_ids = []
                for i in range(len(tokens_a) + 2 + ori_Q_len):
                    position_ids.append(i)
                for i in range(len(tokens_a) + 2 + ori_Q_len, len(tokens)):
                    position_ids.append(0)
                for i in range(len(tokens), self.max_len):
                    position_ids.append(i - len(tokens) + len(tokens_a) + 2 + ori_Q_len)

                    # Token Indexing
                input_ids = self.indexer(tokens)

                # self-attention mask
                num_img = len(gold_feature_paths)
                input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)

                img_end_pos = 1 + self.len_vis_input * num_img
                if self.use_img_content: input_mask[:, :img_end_pos].fill_(1)
                st, end = 1 + self.max_len_img_cxt, len(
                    tokens_a) + 2 + ori_Q_len  # paddings at the end of tokens_b don't need attention
                input_mask[:, st:end].fill_(1)
                # Tokens in A can attend to previous tokens in A
                pred_st, pred_end = len(tokens), self.max_len
                input_mask[pred_st:pred_end, pred_st:pred_end].copy_(
                    self._tril_matrix[:pred_end - pred_st, :pred_end - pred_st])

                # Convert some inputs to tensors
                input_ids = torch.LongTensor(input_ids)
                segment_ids = torch.LongTensor(segment_ids)
                position_ids = torch.LongTensor(position_ids)

                img_list = []
                vis_pe_list = []
                for img_path in gold_feature_paths:
                    assert os.path.exists(img_path), "loader Processor: .pkl file doesn't exist! {}".format(img_path)
                    with open(img_path, "rb") as f:
                        features = TorchCPUUnpickler(f).load()
                    img = features['fc1_features'].detach().cpu().float()
                    cls_label = features['cls_features'].detach().cpu().float()
                    vis_pe = features['pred_boxes'].detach().cpu()

                    # Lazy normalization of the coordinates
                    w_est = torch.max(vis_pe[:, [0, 2]]) * 1. + 1e-5
                    h_est = torch.max(vis_pe[:, [1, 3]]) * 1. + 1e-5
                    vis_pe[:, [0, 2]] /= w_est
                    vis_pe[:, [1, 3]] /= h_est
                    assert h_est > 0, 'loader Processor: box h_est should greater than 0! {}'.format(h_est)
                    assert w_est > 0, 'loader Processor: box w_est should greater than 0! {}'.format(w_est)
                    rel_area = (vis_pe[:, 3] - vis_pe[:, 1]) * (vis_pe[:, 2] - vis_pe[:, 0])
                    rel_area.clamp_(0)

                    vis_pe = torch.cat(
                        (vis_pe[:, :4], rel_area.view(-1, 1), features['scores'].detach().cpu().view(-1, 1)), -1)
                    normalized_coord = F.normalize(vis_pe.data[:, :5] - 0.5, dim=-1)
                    vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), F.layer_norm(cls_label, [1601])), dim=-1)

                    img_list.append(img)
                    vis_pe_list.append(vis_pe)
                    if len(img_list) >= 2: break  # harded coded, doesn't allow more than 2 imgs

                if len(img_list) == 0:
                    assert len(vis_pe_list) == 0
                    img = torch.zeros((self.max_len_img_cxt, 2048))  # 2048 is hard-coded
                    vis_pe = torch.zeros((self.max_len_img_cxt, 1607))  # 1607 is hard-coded
                else:
                    img = torch.cat(img_list, dim=0)
                    vis_pe = torch.cat(vis_pe_list, dim=0)
                    assert img.size(0) == vis_pe.size(0), "img features and vis_pe should have the same token length!"
                    vis_pad = torch.zeros((self.max_len_img_cxt - img.size(0), img.size(-1)))  # .to(device)
                    img = torch.cat((img, vis_pad), dim=0)
                    vis_pad = torch.zeros((self.max_len_img_cxt - vis_pe.size(0), vis_pe.size(-1)))  # .to(device)
                    vis_pe = torch.cat((vis_pe, vis_pad), dim=0)
                assert vis_pe.size(0) == self.max_len_img_cxt
                assert img.size(0) == self.max_len_img_cxt

                cxt_modality_label = [1]
                # previous schema: (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next_label, do_filter_task, filter_label, logit_mask, ori_choices, task_idx, img, vis_pe, context, cxt_modality_label, example_id)
                #          schema: (input_ids, segment_ids, position_ids, input_mask, ----------------------------------------------------------------------------------------------- task_idx, img, vis_pe, context, cxt_modality_label, example_id)
                return {"input_ids": input_ids,
                        "segment_ids": segment_ids,
                        "position_ids": position_ids,
                        "input_mask": input_mask,
                        "task_idx": self.task_idx,
                        "img": img,
                        "vis_pe": vis_pe,
                        "context": context,
                        "cxt_modality_label": cxt_modality_label,
                        "example_id": example_id
                        }


            else:  # qa task, context is txt
                gold_facts, distractor_facts, gold_cxt_list, distractor_cxt_list, Q, A, do_filter_task, context, example_id = instance
                tokens_a = []
                if self.use_txt_fact: tokens_a = sum(gold_facts, [])
                tokens_b = Q.copy()
                truncate_tokens_pair(tokens_a, tokens_b, max_len=self.max_len_a + self.max_len_Q,
                                     max_len_a=self.max_len_a, max_len_b=self.max_len_Q, trunc_seg=self.trunc_seg,
                                     always_truncate_tail=self.always_truncate_tail)

                n_pad = self.max_len_Q + self.max_len_a - len(tokens_a) - len(tokens_b)
                tokens_b += ['[PAD]'] * n_pad

                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b
                if self.new_segment_ids:
                    segment_ids = [4] * (len(tokens_a) + 2) + [5] * len(tokens_b) + [5] * (self.max_len - len(tokens))
                else:
                    segment_ids = [0] * (len(tokens_a) + 2) + [1] * len(tokens_b) + [5] * (self.max_len - len(tokens))

                ori_Q_len = min(len(Q), self.max_len_Q)
                position_ids = []
                for i in range(len(tokens_a) + 2 + ori_Q_len):
                    position_ids.append(i)
                for i in range(len(tokens_a) + 2 + ori_Q_len, len(tokens)):
                    position_ids.append(0)
                for i in range(len(tokens), self.max_len):
                    position_ids.append(i - len(tokens) + len(tokens_a) + 2 + ori_Q_len)

                input_ids = self.indexer(tokens)

                input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
                input_mask[:, :len(tokens_a) + 2 + ori_Q_len].fill_(1)
                pred_st, pred_end = len(tokens), self.max_len
                input_mask[pred_st:pred_end, pred_st:pred_end].copy_(
                    self._tril_matrix[:pred_end - pred_st, :pred_end - pred_st])

                # Convert some inputs to tensors
                input_ids = torch.LongTensor(input_ids)
                segment_ids = torch.LongTensor(segment_ids)
                position_ids = torch.LongTensor(position_ids)

                # schema: (input_ids, segment_ids, position_ids, input_mask, self.task_idx, img, vis_pe, context, cxt_modality_label, example_id)
                return {"input_ids": input_ids,
                        "segment_ids": segment_ids,
                        "position_ids": position_ids,
                        "input_mask": input_mask,
                        "task_idx": self.task_idx,
                        "img": None,
                        "vis_pe": None,
                        "context": context,
                        "cxt_modality_label": None,
                        "example_id": example_id
                        }
