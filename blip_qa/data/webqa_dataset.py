import os
from random import shuffle
import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Literal
from data.utils import pre_question, pre_caption
import torch.nn.functional as F


# TODO
class WebQADataset(Dataset):
    def __init__(
            self, data_json, transform, image_dir, eos='[SEP]', split="train", max_ques_words=30,
            ignored_questions: List[str] = None, use_num_samples: int = -1,
            qcate: Literal['text', 'YesNo', 'Others', 'choose', 'number', 'color', 'shape', 'all'] = 'all',
    ):
        if ignored_questions is None:
            ignored_questions = []
        self.qcate = ['YesNo', 'Others', 'choose', 'number', 'color', 'shape']
        if 'all' not in qcate:
            self.qcate = list(set(qcate).intersection(set(self.qcate)))

        self.split = split
        self.transform = transform
        self.image_dir = image_dir
        self.max_ques_words = max_ques_words
        self.eos = eos

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
                        Q = datum['Q'].replace('"', "")
                        A = datum['A'][0].replace('"', "")

                        gold_text_facts, gold_img_and_caps = self.extract_facts_for_question(datum)
                        if not self.check_image_feature_path(gold_img_and_caps):
                            print(f"Question {i} skipped")
                            continue

                        assert len(gold_text_facts) > 0 or len(gold_img_and_caps) > 0
                        shuffle(gold_text_facts)
                        shuffle(gold_img_and_caps)

                        # ========================
                        self.instance_list.append(
                            (gold_text_facts, gold_img_and_caps, Q, A, question_id)
                        )
                        count += 1

        print(f"Load {len(self.instance_list)} instances from {count} samples")

    def extract_facts_for_question(self, datum: dict):
        # text facts
        gold_text_facts = []
        if 'txt_posFacts' in datum:
            for fa in datum['txt_posFacts']:
                gold_text_facts.append({
                    'fact': fa['fact'],
                    'snippet_id': fa['snippet_id']
                })

        # image facts
        gold_img_and_caps = []
        if 'img_posFacts' in datum:
            for im in datum['img_posFacts']:
                gold_img_and_caps.append(self.load_image_fact(im))

        return gold_text_facts, gold_img_and_caps

    def load_image_fact(self, im: dict):
        image_feature_path = os.path.join(self.image_dir, f"{im['image_id']}.jpg")
        cap = im['caption'].strip()
        return image_feature_path, cap

    def check_image_feature_path(self, facts):
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
        """

        text_facts, img_and_caps, Q, A, question_id = self.instance_list[index]

        # [(channel, width, height), ...]
        images = [self.transform(Image.open(img_path).convert('RGB')) for img_path, _ in img_and_caps]
        captions = [pre_caption(cap, self.max_ques_words) for _, cap in img_and_caps]

        return (
            torch.stack(images),
            captions,
            pre_question(Q, self.max_ques_words),
            pre_question(A, self.max_ques_words),
        )


def webqa_collate_fn(batch):
    """
    :return:
        - image_pad: (batch, n_facts, channel, H, W), with 0 padded to the end of n_facts dimension
        - caption_lists: a batch of list of captions
        - questions: a batch of questions
        - answers: a batch of answers
        - n_facts: a list of integers
    """
    max_n_facts = 0
    image_lists, caption_lists, questions, answers, n_facts = [], [], [], [], []
    for image, caption, question, answer in batch:
        image_lists.append(image)
        max_n_facts = max(max_n_facts, image.size(0))

        n_facts.append(image.size(0))
        caption_lists.append(caption)
        questions.append(question)
        answers.append(answer)

    image_pad = [
        F.pad(img, (0, 0, 0, 0, 0, 0, 0, max_n_facts - img.size(0)))
        for img in image_lists
    ]
    image_pad = torch.stack(image_pad)
    return image_pad, caption_lists, questions, answers, n_facts
