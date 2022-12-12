import os
import random
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import List
from data.utils import pre_caption
import torch.nn.functional as F


class WebQADataset(Dataset):
    def __init__(
            self, data_json, transform, image_dir, eos='[SEP]', split="train",
            ignored_questions: List[str] = None, use_num_samples: int = -1,
            image_only=False, max_n_neg_facts=0, cased=True, no_img_input=False,
    ):
        if ignored_questions is None:
            ignored_questions = []

        if image_only:
            self.qcate = ['YesNo', 'Others', 'choose', 'number', 'color', 'shape']
        else:
            self.qcate = ['YesNo', 'Others', 'choose', 'number', 'color', 'shape', 'text']

        self.qcate2index = {}
        for i, qc in enumerate(self.qcate):
            self.qcate2index[qc] = i

        self.no_img_input = no_img_input
        self.split = split
        self.transform = transform
        self.image_dir = image_dir
        self.eos = eos
        self.max_n_neg_facts = max_n_neg_facts
        self.cased = cased

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
                        instance = self.load_question_instance(datum)
                        if instance is None:
                            continue

                        self.instance_list.append(instance)
                        count += 1

        print(f"Load {len(self.instance_list)} instances from {count} samples")

    def clean_text(self, text: str) -> str:
        return pre_caption(text.replace('"', ""), self.cased, max_words=40)

    def load_question_instance(self, datum: dict):
        question_id = datum['Guid']
        Q = self.clean_text(datum['Q'])
        A = self.clean_text(datum['A'])

        gold_text_facts, neg_text_facts, gold_img_and_caps, neg_img_and_caps = self.extract_sources_for_question(datum)

        if not self.check_image_feature_path(gold_img_and_caps):
            print(f"Question {question_id} skipped because image is not found")
            return None

        assert len(gold_text_facts) > 0 or len(gold_img_and_caps) > 0

        return (
            gold_text_facts, neg_text_facts, gold_img_and_caps, neg_img_and_caps,
            Q, A, question_id, datum['Qcate'] if self.split != 'test' else '',
        )

    def extract_sources_for_question(self, datum: dict):
        qid = datum['Guid']
        # text
        gold_text_facts = []
        neg_text_facts = []
        if self.split == 'test':
            gold_text_facts += self.load_text_facts(qid, datum['txt_Facts'])
        else:
            if 'txt_posFacts' in datum:
                gold_text_facts += self.load_text_facts(qid, datum['txt_posFacts'])
            if 'txt_negFacts' in datum:
                neg_text_facts += self.load_text_facts(qid, datum['txt_negFacts'])

        # image
        gold_img_and_caps = []
        neg_img_and_caps = []
        if self.split == 'test':
            gold_img_and_caps += self.load_image_facts(qid, datum['img_Facts'])
        else:
            if 'img_posFacts' in datum:
                gold_img_and_caps += self.load_image_facts(qid, datum['img_posFacts'])
            if 'img_negFacts' in datum:
                neg_img_and_caps += self.load_image_facts(qid, datum['img_negFacts'])

        return gold_text_facts, neg_text_facts, gold_img_and_caps, neg_img_and_caps

    def load_text_facts(self, qid: str, txt: List[dict]):
        return [self.clean_text(t['fact']) for t in txt]

    def load_image_facts(self, qid: str, im: List[dict]):
        return [
            (
                os.path.join(self.image_dir, f"{m['image_id']}.jpg"),
                self.clean_text(m['caption'].strip()),
            ) for m in im
        ]

    @staticmethod
    def check_image_feature_path(facts):
        for path, _ in facts:
            if not os.path.exists(path):
                print(f'Cannot find image at {path}')
                return False
        return True

    def __len__(self):
        return len(self.instance_list)

    def get_item_from_instance(self, instance):
        text, neg_text, img_caps, neg_img_caps, Q, A, question_id, qcate = instance

        if self.split != 'test':
            n_neg_facts = random.randint(0, self.max_n_neg_facts)
            if len(neg_img_caps) > n_neg_facts:
                neg_img_caps = random.sample(neg_img_caps, n_neg_facts)
        else:
            neg_img_caps = []

        all_text = text
        all_img_caps = img_caps + neg_img_caps
        img_retr_tgts = torch.cat(
            [torch.ones(len(img_caps)), torch.zeros(len(neg_img_caps))]
        )

        all_img_caps, img_retr_tgts = self.shuffle_facts_and_retr_labels(all_img_caps, img_retr_tgts)

        # pos/neg captions + pos/neg text facts
        captions = [cap for _, cap in all_img_caps]
        captions += all_text
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

    def __getitem__(self, index):
        """
        :return:
            images: (n_facts, channel, H, W)
            captions: a list of strings
            questions: a list of strings
            answers: a list of strings
            retrieval_labels: list of 1 or 0s, 1 means gold facts, 0 means negative/distractor facts
        """
        return self.get_item_from_instance(self.instance_list[index])

    @staticmethod
    def shuffle_facts_and_retr_labels(facts: list, retr_labels: torch.Tensor):
        shuff_idx = list(range(len(facts)))
        random.shuffle(shuff_idx)
        facts = [facts[i] for i in shuff_idx]
        retr_labels = retr_labels[shuff_idx]
        return facts, retr_labels

    def collate_fn(self, batch):
        """
        :return:
            - image_pad: (batch, n_facts, channel, H, W), with 0 padded to the end of n_facts dimension
            - caption_lists: a batch of list of captions
            - questions: a batch of questions
            - answers: a batch of answers
            - n_facts: a list of integers
            - retr_labels: a batch of list of 1 and 0
        """
        pad_max_len = 0
        (
            image_lists, caption_lists, questions, answers, n_facts, question_ids, qcates, retr_labels
        ) = [], [], [], [], [], [], [], []
        for image, caption, question, answer, qid, qcate, retr in batch:
            if image is None:  # placeholder for samples without image facts
                image_lists.append(torch.zeros(1, 3, 480, 480))
                n_facts.append(0)  # set to 0 so the placeholder is masked
                pad_max_len = max(pad_max_len, 1)
            else:
                image_lists.append(image)
                if self.no_img_input:
                    n_facts.append(0)
                else:
                    n_facts.append(image.size(0))
                pad_max_len = max(pad_max_len, image.size(0))

            caption_lists.append(caption)
            questions.append(question)
            answers.append(answer)

            question_ids.append(qid)
            qcates.append(qcate)

            retr_labels.append(retr)

        image_pad = [
            F.pad(img, (0, 0, 0, 0, 0, 0, 0, pad_max_len - img.size(0)))
            for img in image_lists
        ]
        image_pad = torch.stack(image_pad)

        return image_pad, caption_lists, questions, answers, n_facts, question_ids, qcates, retr_labels


class WebQADatasetWithOFARankingScore(WebQADataset):
    def __init__(self, ranking_score_json: str, *args, **kwargs):
        with open(ranking_score_json, encoding='utf-8') as f:
            self.ranking_scores = json.load(f)
        for qid in self.ranking_scores.items():
            data = self.ranking_scores[qid]
            src_scores = {}
            for src in data['sources']:
                src_scores[src[0]] = src[1]
            self.ranking_scores[qid] = src_scores

        super().__init__(*args, **kwargs)

    def load_text_facts(self, qid: str, txt: List[dict]):
        return [
            (
                self.clean_text(t['fact']),  # text
                self.ranking_scores[qid][t['snippet_id']],  # ranking score
            ) for t in txt
            if t['snippet_id'] in self.ranking_scores[qid]
        ]

    def load_image_facts(self, qid: str, im: List[dict]):
        return [
            (
                os.path.join(self.image_dir, f"{m['image_id']}.jpg"),  # image path
                self.clean_text(m['caption'].strip()),  # image caption
                self.ranking_scores[qid][m['image_id']],  # ranking score
            ) for m in im
            if m['image_id'] in self.ranking_scores[qid]
        ]

    @staticmethod
    def check_image_feature_path(facts):
        for path, _ in facts:
            if not os.path.exists(path):
                print(f'Cannot find image at {path}')
                return False
        return True

    def get_item_from_instance(self, instance):
        ret = super().get_item_from_instance(instance)
        img_caps, neg_img_caps = instance[2], instance[3]
        ret += (img_ranking_scores,)

    def collate_fn(self, batch):
        """
        :return:
            - image_pad: (batch, n_facts, channel, H, W), with 0 padded to the end of n_facts dimension
            - caption_lists: a batch of list of captions
            - questions: a batch of questions
            - answers: a batch of answers
            - n_facts: a list of integers
            - retr_labels: a batch of list of 1 and 0
            - ranking_scores: a batch of tensors with padded ranking scores
        """
        pad_max_len = 0
        (
            image_lists, caption_lists, questions, answers, n_facts, question_ids, qcates, retr_labels
        ) = [], [], [], [], [], [], [], []
        for image, caption, question, answer, qid, qcate, retr in batch:
            if image is None:  # placeholder for samples without image facts
                image_lists.append(torch.zeros(1, 3, 480, 480))
                n_facts.append(0)  # set to 0 so the placeholder is masked
                pad_max_len = max(pad_max_len, 1)
            else:
                image_lists.append(image)
                if self.no_img_input:
                    n_facts.append(0)
                else:
                    n_facts.append(image.size(0))
                pad_max_len = max(pad_max_len, image.size(0))

            caption_lists.append(caption)
            questions.append(question)
            answers.append(answer)

            question_ids.append(qid)
            qcates.append(qcate)

            retr_labels.append(retr)

        image_pad = [
            F.pad(img, (0, 0, 0, 0, 0, 0, 0, pad_max_len - img.size(0)))
            for img in image_lists
        ]
        image_pad = torch.stack(image_pad)

        return image_pad, caption_lists, questions, answers, n_facts, question_ids, qcates, retr_labels
