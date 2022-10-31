from random import shuffle
import pickle
import json
from torch.utils.data import Dataset
from vlp.loader_utils import get_image_feature_path
from typing import Literal

ignored_questions = [
    # RCNN features are missing
    "d5bca9220dba11ecb1e81171463288e9",
    "d5bd79ec0dba11ecb1e81171463288e9",
    "d5c1b5ca0dba11ecb1e81171463288e9",
    "d5c35e5c0dba11ecb1e81171463288e9",
    "d5c404b00dba11ecb1e81171463288e9",
    "d5c6bc5a0dba11ecb1e81171463288e9",
    "d5c73d240dba11ecb1e81171463288e9",
    "d5c7f5660dba11ecb1e81171463288e9",
    "d5c81ff00dba11ecb1e81171463288e9",
    "d5c88ed60dba11ecb1e81171463288e9",
    "d5c8e70a0dba11ecb1e81171463288e9",
    "d5c9de120dba11ecb1e81171463288e9",
    "d5ca437a0dba11ecb1e81171463288e9",
    "d5cb80320dba11ecb1e81171463288e9",
    "d5d0a2ce0dba11ecb1e81171463288e9",
    "d5d214600dba11ecb1e81171463288e9",
    "d5d5fcec0dba11ecb1e81171463288e9",
    "d5d72c0c0dba11ecb1e81171463288e9",
    "d5d758d00dba11ecb1e81171463288e9",
    "d5d8e88a0dba11ecb1e81171463288e9",
    "d5d9492e0dba11ecb1e81171463288e9",
    "d5dc3f3a0dba11ecb1e81171463288e9",
    "d5de14720dba11ecb1e81171463288e9",
    "d5de1b340dba11ecb1e81171463288e9",

    # invalid facts
    "d5d84d1c0dba11ecb1e81171463288e9",
]


class WebQARetrievalDataset(Dataset):
    def __init__(
            self,
            dataset_json_path: str,
            split: Literal['train', 'test', 'val'],
            Qcate: Literal['text', 'YesNo', 'Others', 'choose', 'number', 'color', 'shape'],
            batch_size: int,
            tokenizer, feature_folder: str,
            use_num_samples: int,
            processor,
            answer_provided_by: Literal['img', 'txt'],
            use_x_distractors=True,
            max_snippets=10,
            max_imgs=10,
            imgid_map=None,
            device=None,
    ):
        """
        :param dataset_json_path: Data json file path
        :param split: Data split
        :param Qcate: Question category
        :param batch_size: Batch size
        :param tokenizer: Bert tokenizer
        :param feature_folder: Image feature folder
        :param use_num_samples: Max number of questions to use, remaining data is not loaded
        :param processor: Post-processor
        :param answer_provided_by: What modality the positive fact has
        :param use_x_distractors: Whether to use cross-modality distractors (negative facts). In other words, if
                answer_is_provided_by == 'txt', the distractors are images
        :param max_snippets: Max number of text facts
        :param max_imgs: Max number of image facts
        :param imgid_map: Image id to feature id mapping
        :param device: PyTorch device
        """
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.answer_provided_by = answer_provided_by
        self.max_snippets = max_snippets
        self.max_imgs = max_imgs
        self.instance_list = []
        self.feature_folder = feature_folder
        self.use_x_distractors = use_x_distractors

        # Qcate == text question contain text positive and text+image negative facts
        # Otherwise contains image positive and text+image negative facts

        if answer_provided_by == 'txt':
            self.Qcate = ['text']
        else:
            self.Qcate = ['YesNo', 'Others', 'choose', 'number', 'color', 'shape']
            if 'all' not in Qcate:
                self.Qcate = list(set(Qcate).intersection(set(self.Qcate)))

        if imgid_map is not None:
            self.imgid_map = pickle.load(open(imgid_map, "rb"))
            print("\nLoad imgid_map, length={}\n".format(len(self.imgid_map)))
        else:
            self.imgid_map = None

        if device is not None:
            self.device = device

        with open(dataset_json_path, "r") as f:
            dataset = json.load(f)

        assert answer_provided_by in ['img', 'txt']
        modality_type = answer_provided_by
        if use_x_distractors:
            modality_type = 'both'

        # load all samples
        count = 0
        for i, datum in dataset.items():
            if i in ignored_questions:
                continue
            data_split = datum['split']
            if data_split in split:
                if data_split == 'test' or datum['Qcate'] in self.Qcate:
                    if use_num_samples == -1 or count < use_num_samples:
                        Guid = datum['Guid']
                        Q = self.tokenizer.tokenize(datum['Q'].replace('"', ""))
                        A = self.tokenizer.tokenize(datum['A'][0].replace('"', ""))

                        (
                            gold_text_facts, distractor_text_facts, gold_img_and_caps, distractor_img_and_caps
                        ) = (
                            self.extract_facts_for_question_test(datum) if datum['split'] == 'test'
                            else self.extract_facts_for_question(datum)
                        )

                        shuffle(gold_text_facts)
                        shuffle(distractor_text_facts)
                        shuffle(gold_img_and_caps)
                        shuffle(distractor_img_and_caps)

                        # at least one type of gold fact and one type of distractor fact
                        assert len(gold_text_facts) > 0 or len(gold_img_and_caps) > 0
                        if data_split != 'test':
                            assert len(distractor_text_facts) > 0 or len(distractor_img_and_caps) > 0

                        # ========================
                        self.instance_list.append(
                            (gold_text_facts, distractor_text_facts, gold_img_and_caps,
                             distractor_img_and_caps, Q, A, True, modality_type,
                             Guid)
                        )

                        count += 1

        print(f"Load {len(self.instance_list)} instances from {count} samples")

    def extract_facts_for_question(self, datum: dict):
        # text facts
        gold_text_facts = []
        distractor_text_facts = []
        if 'txt_posFacts' in datum:
            for fa in datum['txt_posFacts']:
                gold_text_facts.append({
                    'fact': self.tokenizer.tokenize(fa['fact']),
                    'snippet_id': fa['snippet_id']
                })
        if 'txt_negFacts' in datum:
            for fa in datum['txt_negFacts']:
                distractor_text_facts.append({
                    'fact': self.tokenizer.tokenize(fa['fact']),
                    'snippet_id': fa['snippet_id']
                })

        # image facts
        gold_img_and_caps = []
        distractor_img_and_caps = []
        if 'img_posFacts' in datum:
            for im in datum['img_posFacts']:
                gold_img_and_caps.append(self.load_image_fact(im))
        if 'img_negFacts' in datum:
            for im in datum['img_negFacts']:
                distractor_img_and_caps.append(self.load_image_fact(im))

        return gold_text_facts, distractor_text_facts, gold_img_and_caps, distractor_img_and_caps

    def extract_facts_for_question_test(self, datum: dict):
        # text facts
        gold_text_facts = []
        for fa in datum['txt_Facts']:
            gold_text_facts.append({
                'fact': self.tokenizer.tokenize(fa['fact']),
                'snippet_id': fa['snippet_id']
            })

        # image facts
        gold_img_and_caps = []
        for im in datum['img_Facts']:
            gold_img_and_caps.append(self.load_image_fact(im))

        return gold_text_facts, [], gold_img_and_caps, []

    def load_image_fact(self, im: dict):
        image_id = int(im['image_id'])
        if self.imgid_map is not None:
            image_id = self.imgid_map[image_id]
        image_feature_path = get_image_feature_path(self.feature_folder, image_id)

        cap = self.tokenizer.tokenize(im['caption'].strip())
        return image_feature_path, cap

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, idx):
        (
            gold_text_facts, distractor_text_facts, gold_img_and_caps, distractor_img_and_caps,
            Q, A, do_filter_task, context, example_id,
        ) = self.instance_list[idx]

        # make sure number of gold+negative facts is <= self.max_imgs or self.max_snippets
        # if not, remove extra negative facts, keeping all positive facts
        # txt
        distractor_text_facts = distractor_text_facts[:self.max_snippets - len(gold_text_facts)]
        # img
        distractor_img_and_caps = distractor_img_and_caps[:self.max_imgs - len(gold_img_and_caps)]

        # NOTE: for test split, all facts are kept in gold_xxx_facts
        # Therefore, the number of facts might be bigger than self.max_snippets + self.max_imgs

        # postprocess
        instance = (
            gold_text_facts, distractor_text_facts, gold_img_and_caps, distractor_img_and_caps,
            Q, A, do_filter_task, context, example_id
        )
        return self.processor(instance, self.max_imgs + self.max_snippets, self.device)
