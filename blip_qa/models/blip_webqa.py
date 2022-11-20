from typing import List
from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError(f"length_dim cannot be 0: {length_dim}")

    if not isinstance(lengths, list):
        lengths = lengths.long().tolist()

    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


class BLIP_VQA(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=480,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer,
                                                       drop_path_rate=0.1)
        self.tokenizer = init_tokenizer()

        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)

        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

    def encode_images(self, images: torch.Tensor, n_facts: List[int]):
        """
        :param images: (batch, n_facts, channel, H, W)
        :param n_facts: Number of image facts in each sample in the batch
        :return:
            - image_embeds: (batch, n_facts * seq_len, embed_size)
            - lengths: Valid lengths (dim 1) of image_embeds
        """
        batch_size, max_n_facts, C, H, W = images.shape
        images = images.view(-1, C, H, W)
        image_embeds = self.visual_encoder(images)  # (batch * n_facts, seq_len, embed_size)

        n_facts = [nf * image_embeds.size(1) for nf in n_facts]
        image_embeds = image_embeds.view(batch_size, max_n_facts * image_embeds.size(1), image_embeds.size(2))
        return image_embeds, n_facts

    def forward(
            self,
            image: torch.Tensor,
            captions: List[List[str]],
            question: List[str],
            answer: List[str],
            n_facts: List[int],
            train=True,
            inference='rank',
            k_test=128,
    ):
        """
        :param image: (batch, n_facts, channel, H, W)
        :param captions: Batch of list of captions
        :param question: Batch of questions
        :param answer: Batch of answers
        :param n_facts: Batch of number of image facts
        :param train: train or inference
        :param inference: inference mode
        :param k_test: top_k setting
        :return:
        """

        image_embeds, lengths = self.encode_images(image, n_facts)
        image_atts = make_pad_mask(lengths, image_embeds[:, :, 0], 1).to(image.device)

        question = self.tokenizer(question, padding='longest',  # truncation=True, max_length=35,
                                  return_tensors="pt").to(image.device)
        question.input_ids[:, 0] = self.tokenizer.enc_token_id

        # concatenate captions and tokenize them
        captions = [' '.join(cap) for cap in captions]
        captions = self.tokenizer(
            captions, padding='longest', return_tensors="pt"
        ).to(image.device)
        captions.input_ids[:, 0] = self.tokenizer.sep_token_id

        if train:
            '''
            n: number of answers for each question
            weights: weight for each answer
            '''
            answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device)
            answer.input_ids[:, 0] = self.tokenizer.bos_token_id
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

            question_output = self.text_encoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)

            answer_output = self.text_decoder(answer.input_ids,
                                              attention_mask=answer.attention_mask,
                                              encoder_hidden_states=question_output.last_hidden_state,
                                              encoder_attention_mask=question.attention_mask,
                                              labels=answer_targets,
                                              return_dict=True,
                                              reduction='none',
                                              )

            loss = answer_output.loss
            loss = loss.sum() / image.size(0)
            return loss

        else:
            question_output = self.text_encoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)

            # FIXME:
            if inference == 'generate':
                num_beams = 3
                question_states = question_output.last_hidden_state.repeat_interleave(num_beams, dim=0)
                question_atts = torch.ones(question_states.size()[:-1], dtype=torch.long).to(question_states.device)
                model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask": question_atts}

                bos_ids = torch.full((image.size(0), 1), fill_value=self.tokenizer.bos_token_id, device=image.device)

                outputs = self.text_decoder.generate(input_ids=bos_ids,
                                                     max_length=10,
                                                     min_length=1,
                                                     num_beams=num_beams,
                                                     eos_token_id=self.tokenizer.sep_token_id,
                                                     pad_token_id=self.tokenizer.pad_token_id,
                                                     **model_kwargs)

                answers = []
                for output in outputs:
                    answer = self.tokenizer.decode(output, skip_special_tokens=True)
                    answers.append(answer)
                return answers

            elif inference == 'rank':
                max_ids = self.rank_answer(question_output.last_hidden_state, question.attention_mask,
                                           answer.input_ids, answer.attention_mask, k_test)
                return max_ids

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):

        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states=question_states,
                                         encoder_attention_mask=question_atts,
                                         return_dict=True,
                                         reduction='none')
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(input_ids,
                                   attention_mask=input_atts,
                                   encoder_hidden_states=question_states,
                                   encoder_attention_mask=question_atts,
                                   labels=targets_ids,
                                   return_dict=True,
                                   reduction='none')

        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques, k)

        max_topk_ids = log_probs_sum.argmax(dim=1)
        max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]

        return max_ids


def blip_vqa(pretrained='', **kwargs):
    model = BLIP_VQA(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    return model


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))
