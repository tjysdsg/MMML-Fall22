import os
import random
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Literal
from data.utils import pre_caption
import torch.nn.functional as F


class WebQADataset(Dataset):
    def __init__(
            self, data_json, transform, image_dir, eos='[SEP]', split="train",
            ignored_questions: List[str] = None, use_num_samples: int = -1,
            qcate: Literal['text', 'YesNo', 'Others', 'choose', 'number', 'color', 'shape', 'all'] = 'all',
            max_n_neg_facts=4,
    ):
        if ignored_questions is None:
            ignored_questions = []
        self.qcate = ['YesNo', 'Others', 'choose', 'number', 'color', 'shape', 'text']
        if 'all' not in qcate:
            self.qcate = list(set(qcate).intersection(set(self.qcate)))

        self.split = split
        self.transform = transform
        self.image_dir = image_dir
        self.eos = eos
        self.max_n_neg_facts = max_n_neg_facts

        self.instance_list = []
        count = 0
        with open(data_json, "r", encoding='utf-8') as f:
            dataset = json.load(f)
        for i, datum in dataset.items():
            if i in ignored_questions:
                continue
            data_split = datum['split']
            if data_split in split:
                if data_split == 'test' or datum['Qcate'] in self.qcate:
                    if use_num_samples == -1 or count < use_num_samples:
                        question_id = datum['Guid']
                        Q = pre_caption(datum['Q'].replace('"', ""), 50)
                        A = pre_caption(datum['A'][0].replace('"', ""), 50)

                        gold_text_facts, neg_text_facts = self.extract_text_facts_for_question(datum)
                        gold_img_and_caps, neg_img_and_caps = self.extract_img_facts_for_question(datum)

                        if not self.check_image_feature_path(gold_img_and_caps):
                            print(f"Question {i} skipped because image is not found")
                            continue

                        assert len(gold_text_facts) > 0 or len(gold_img_and_caps) > 0

                        # ========================
                        self.instance_list.append(
                            (
                                gold_text_facts, neg_text_facts, gold_img_and_caps, neg_img_and_caps,
                                Q, A, question_id, datum['Qcate'] if self.split != 'test' else '',
                            )
                        )
                        count += 1

        print(f"Load {len(self.instance_list)} instances from {count} samples")

    def extract_text_facts_for_question(self, datum: dict):
        gold_text_facts = []
        neg_text_facts = []

        if self.split == 'test':
            for fa in datum['txt_Facts']:
                gold_text_facts.append(pre_caption(fa['fact'], 40))
        else:
            if 'txt_posFacts' in datum:
                for fa in datum['txt_posFacts']:
                    gold_text_facts.append(pre_caption(fa['fact'], 40))
            if 'txt_negFacts' in datum:
                for fa in datum['txt_negFacts']:
                    neg_text_facts.append(pre_caption(fa['fact'], 40))
        return gold_text_facts, neg_text_facts

    def extract_img_facts_for_question(self, datum: dict):
        gold_img_and_caps = []
        neg_img_and_caps = []

        if self.split == 'test':
            for im in datum['img_Facts']:
                gold_img_and_caps.append(self.load_image_fact(im))
        else:
            if 'img_posFacts' in datum:
                for im in datum['img_posFacts']:
                    gold_img_and_caps.append(self.load_image_fact(im))
            if 'img_negFacts' in datum:
                for im in datum['img_negFacts']:
                    neg_img_and_caps.append(self.load_image_fact(im))

        return gold_img_and_caps, neg_img_and_caps

    def load_image_fact(self, im: dict):
        image_feature_path = os.path.join(self.image_dir, f"{im['image_id']}.jpg")
        cap = im['caption'].strip()
        return image_feature_path, pre_caption(cap, 100)

    @staticmethod
    def check_image_feature_path(facts):
        for path, _ in facts:
            if not os.path.exists(path):
                print(f'Cannot find image at {path}')
                return False
        return True

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, index):
        """
        :return:
            images: (n_facts, channel, H, W)
            captions: a list of strings
            questions: a list of strings
            answers: a list of strings
            retrieval_labels: list of 1 or 0s, 1 means gold facts, 0 means negative/distractor facts
        """
        text, neg_text, img_caps, neg_img_caps, Q, A, question_id, qcate = self.instance_list[index]

        n_neg_facts = random.randint(0, self.max_n_neg_facts)
        if self.split != 'test':
            if len(neg_img_caps) > n_neg_facts:
                neg_img_caps = random.sample(neg_img_caps, n_neg_facts)
        else:
            neg_img_caps = []

        # TODO: neg_text = random.sample(neg_text, n_neg_facts)
        #   all_text = text + neg_text
        all_text = text
        all_img_caps = img_caps + neg_img_caps
        img_retr_tgts = torch.cat(
            [torch.ones(len(img_caps)), torch.zeros(len(neg_img_caps))]
        )

        # shuffle facts
        # TODO: text_retr_tgts = torch.cat(
        #       [torch.ones(len(text)), torch.zeros(len(neg_text))]
        #   )
        #   all_text, text_retr_tgts  = self.shuffle_facts_and_retr_labels(all_text, text_retr_tgts)
        all_img_caps, img_retr_tgts = self.shuffle_facts_and_retr_labels(all_img_caps, img_retr_tgts)

        # pos/neg captions + pos/neg text facts
        captions = [cap for _, cap in all_img_caps]
        captions += all_text
        # TODO: retrieval_labels = torch.cat([img_retr_tgts, text_retr_tgts])
        retrieval_labels = img_retr_tgts

        # [(channel, width, height), ...]
        images = [self.transform(Image.open(img_path).convert('RGB')) for img_path, _ in all_img_caps]
        images = torch.stack(images) if len(images) > 0 else None

        return (
            images,
            captions,
            Q,
            A,
            question_id,
            qcate,
            retrieval_labels,
        )

    @staticmethod
    def shuffle_facts_and_retr_labels(facts: list, retr_labels: torch.Tensor):
        shuff_idx = list(range(len(facts)))
        random.shuffle(shuff_idx)
        facts = [facts[i] for i in shuff_idx]
        retr_labels = retr_labels[shuff_idx]
        return facts, retr_labels


def webqa_collate_fn(batch):
    """
    :return:
        - image_pad: (batch, n_facts, channel, H, W), with 0 padded to the end of n_facts dimension
        - caption_lists: a batch of list of captions
        - questions: a batch of questions
        - answers: a batch of answers
        - n_facts: a list of integers

    NOTE: Only support batch size of 1 because for text-based questions, we don't use cross attention,
    which means we cannot mix text-based and image-based questions in a batch
    """
    pad_max_len = 0
    (
        image_lists, caption_lists, questions, answers, n_facts, question_ids, qcates, retr_labels
    ) = [], [], [], [], [], [], [], []
    for image, caption, question, answer, qid, qcate, retr in batch:
        if image is None:  # placeholder for samples without image facts
            image_lists.append(None)
            n_facts.append(0)  # set to 0 so the placeholder is masked
            pad_max_len = max(pad_max_len, 1)
        else:
            image_lists.append(image)
            n_facts.append(image.size(0))
            pad_max_len = max(pad_max_len, image.size(0))

        caption_lists.append(caption)
        questions.append(question)
        answers.append(answer)

        question_ids.append(qid)
        qcates.append(qcate)

        retr_labels.append(retr)

    if image_lists[0] is not None:
        image_pad = [
            F.pad(img, (0, 0, 0, 0, 0, 0, 0, pad_max_len - img.size(0)))
            for img in image_lists
        ]
        image_pad = torch.stack(image_pad)
    else:
        image_pad = torch.zeros(1)

    return image_pad, caption_lists, questions, answers, n_facts, question_ids, qcates, retr_labels
