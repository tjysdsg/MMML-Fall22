import os
from random import shuffle
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
                        Q = pre_caption(datum['Q'].replace('"', ""), 100)
                        A = pre_caption(datum['A'][0].replace('"', ""), 100)

                        gold_text_facts, gold_img_and_caps = [], []
                        if 'txt_posFacts' in datum:
                            gold_text_facts = self.extract_text_facts_for_question(datum)
                        if 'img_posFacts' in datum:
                            gold_img_and_caps = self.extract_img_facts_for_question(datum)

                        if not self.check_image_feature_path(gold_img_and_caps):
                            print(f"Question {i} skipped because image is not found")
                            continue

                        assert len(gold_text_facts) > 0 or len(gold_img_and_caps) > 0
                        shuffle(gold_text_facts)
                        shuffle(gold_img_and_caps)

                        # ========================
                        self.instance_list.append(
                            (gold_text_facts, gold_img_and_caps, Q, A, question_id, datum['Qcate'])
                        )
                        count += 1

        print(f"Load {len(self.instance_list)} instances from {count} samples")

    @staticmethod
    def extract_text_facts_for_question(datum: dict):
        gold_text_facts = []
        assert 'txt_posFacts' in datum

        for fa in datum['txt_posFacts']:
            gold_text_facts.append(
                pre_caption(fa['fact'], 100),
            )
        return gold_text_facts

    def extract_img_facts_for_question(self, datum: dict):
        gold_img_and_caps = []
        assert 'img_posFacts' in datum

        for im in datum['img_posFacts']:
            gold_img_and_caps.append(
                self.load_image_fact(im)
            )

        return gold_img_and_caps

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
        """
        text_facts, img_and_caps, Q, A, question_id, qcate = self.instance_list[index]

        # [(channel, width, height), ...]
        images = [self.transform(Image.open(img_path).convert('RGB')) for img_path, _ in img_and_caps]
        captions = [cap for _, cap in img_and_caps]
        captions += text_facts

        images = torch.stack(images) if len(images) > 0 else None
        return (
            images,
            captions,
            Q,
            A,
            question_id,
            qcate,
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
    pad_max_len = 0
    image_lists, caption_lists, questions, answers, n_facts, question_ids, qcates = [], [], [], [], [], [], []
    for image, caption, question, answer, qid, qcate in batch:
        if image is None:  # placeholder for samples without image facts
            image_lists.append(torch.zeros(1, 3, 480, 480))  # FIXME: load H and W from configs
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

    image_pad = [
        F.pad(img, (0, 0, 0, 0, 0, 0, 0, pad_max_len - img.size(0)))
        for img in image_lists
    ]
    image_pad = torch.stack(image_pad)
    return image_pad, caption_lists, questions, answers, n_facts, question_ids, qcates
