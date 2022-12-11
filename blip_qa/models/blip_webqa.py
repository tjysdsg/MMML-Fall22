from typing import List
import torch
from torch import nn
import numpy as np
# from bottleneck_attention import AB


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
                 cased=True,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 multitask_qcate=True,
                 # TODO: add use bottleneck api
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        from models.med import BertConfig, BertModel, BertLMHeadModel
        from models.blip import create_vit, init_tokenizer

        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer,
                                                       drop_path_rate=0.1)
        self.tokenizer = init_tokenizer(cased)
        self.use_bottleneck = True # TODO: change

        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=True)

        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

        self.num_heads = encoder_config.num_attention_heads
        self.num_patches = self.visual_encoder.patch_embed.num_patches + 1
        self.multitask_qcate = multitask_qcate
        if multitask_qcate:
            # self.retr_ffn = nn.Linear(self.num_patches, 1)
            self.multitask_ffn = nn.Linear(encoder_config.hidden_size, 6)

    def encode_images(self, images: torch.Tensor, n_facts: List[int]):
        """
        :param images: (batch, n_facts, channel, H, W)
        :param n_facts: Number of image facts in each sample in the batch
        :return:
            - image_embeds: (batch, n_facts * seq_len, embed_size)
            - lengths: Valid lengths (dim 1) of image_embeds
        """
        # batch_size, max_n_facts, C, H, W = images.shape
        # images = images.view(-1, C, H, W)
        image_embeds = self.visual_encoder(images)  # (batch * n_facts, seq_len, embed_size)

        n_facts = [nf * image_embeds.size(1) for nf in n_facts]
        image_embeds = image_embeds.view(batch_size, max_n_facts * image_embeds.size(1), image_embeds.size(2))
        return image_embeds, n_facts

    def enable_med(self):
        tie_encoder_decoder_weights(self.text_encoder, self.text_decoder.bert, '', '/attention')

    def forward(
            self,
            image: torch.Tensor,
            captions: List[List[str]],
            question: List[str],
            answer: List[str],
            n_img_facts: List[int],
            train=True,
    ):
        """
        :param image: (batch, n_img_facts, channel, H, W)
        :param captions: Batch of list of captions
        :param question: Batch of questions
        :param answer: Batch of answers
        :param n_img_facts: Batch of number of image facts
        :param train: train or inference
        """
        batch_size, max_n_facts, C, H, W = images.shape
        image = images.view(-1, C, H, W)
        if not self.use_bottleneck:
            image_embeds = self.visual_encoder(image, )

        image_embeds, lengths = self.encode_images(image, n_img_facts)
        image_atts = ~make_pad_mask(lengths, image_embeds[:, :, 0], 1).to(image.device)

        question = self.tokenizer(question, padding='longest', return_tensors="pt").to(image.device)
        question.input_ids[:, 0] = self.tokenizer.enc_token_id

        # concatenate captions and tokenize them
        captions = [f' {self.tokenizer.sep_token} '.join(cap) for cap in captions]
        captions = self.tokenizer(captions, padding='longest', return_tensors="pt").to(image.device)
        # mask the first token since we already have a sep_token_id set to the last token of question
        captions.input_ids[:, 0] = self.tokenizer.pad_token_id
        captions.attention_mask[:, 0] = 0

        # image-grounded text encoder
        input_ids = torch.cat([question.input_ids, captions.input_ids], dim=-1)
        attention_mask = torch.cat([question.attention_mask, captions.attention_mask], dim=-1)
        input_ids = input_ids[:, :512]
        attention_mask = attention_mask[:, :512]

        cross_attention_weight = torch.ones_like(input_ids, dtype=torch.float)
        cross_attention_weight[torch.as_tensor(n_img_facts, dtype=torch.long) == 0] = 0.0
        
        # TODO: add use_bottleneck as a flag. refer to 
        https://github.com/google-research/scenic/blob/556bc5be8452560228fa8318f61e414114abfb40/scenic/projects/mbt/model.py#L330

        question_output = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            cross_attention_weight=cross_attention_weight,
            # output_attentions=True,
            return_dict=True,
        )

        def forward():
            # ab = AB()

            vit_layers = self.visual_encoder.blocks
            text_layers = self.text_encoder.encoder.layer
            bottleneck = None
            n_bottlenecks = 4

            bottleneck = None
            if self.use_bottleneck:
                n_bottlenecks = self.n_bottlenecks
            if self.classifier in ['token']:
                n_bottlenecks += 1
            bottleneck = self.param('bottleneck',
                                nn.initializers.normal(stddev=0.02),  # From BERT.
                                (1, n_bottlenecks, c), bottleneck_dtype)
            bottleneck = jnp.tile(bottleneck, [n, 1, 1])


            for i in range(num_layers):
                
                v_in = concact(image, bottleneck)
                v_out = vit_layers[i](v_in)

                t_in = concact(input_ids, bottleneck)
                t_out = text_layers[i](t_in)

                # bottleneck = jnp.mean(jnp.stack(bottle, axis=-1), axis=-1)
                bottleneck = mean(v_out, t_out)

                # v_att = vit_layers[i](ab_images)
                # t_att = text_layers[i].get_attention(input_ids, attention_mask)

                # # bottleneck = ab(v_att, t_att, bottleneck=bottleneck) # (B, SeqLen, nHead, ab_dim)


        # (batch, num_heads, question_len, image_embeds_len)
        multimodal_cross_atts = None
        if train:
            if self.multitask_qcate:
                # multimodal_cross_atts = question_output.cross_attentions[-1]  # last layer's cross attention
                # atts = torch.sum(multimodal_cross_atts, dim=2)  # (batch, num_heads, image_embeds_len)
                # atts = torch.sum(atts, dim=1)  # (batch, image_embeds_len)

                # # (batch, n_facts, num_patches)
                # atts = atts.view(atts.shape[0], -1, self.num_patches)
                # retr = self.retr_ffn(atts).squeeze(dim=-1)  # (batch, n_facts)

                mt_res = self.multitask_ffn(question_output.pooler_output)
            else:
                mt_res = None

            '''
            n: number of answers for each question
            weights: weight for each answer
            '''
            answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device)
            answer.input_ids[:, 0] = self.tokenizer.bos_token_id
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

            answer_output = self.text_decoder(answer.input_ids,
                                              attention_mask=answer.attention_mask,
                                              encoder_hidden_states=question_output.last_hidden_state,
                                              encoder_attention_mask=attention_mask,
                                              labels=answer_targets,
                                              return_dict=True,
                                              reduction='none',
                                              )

            loss = answer_output.loss
            loss = loss.sum() / image.size(0)
            return loss, mt_res, multimodal_cross_atts
        else:
            num_beams = 10
            question_states = question_output.last_hidden_state.repeat_interleave(num_beams, dim=0)
            question_atts = attention_mask.repeat_interleave(num_beams, dim=0).to(question_states.device)

            bos_ids = torch.full((image.size(0), 1), fill_value=self.tokenizer.bos_token_id, device=image.device)

            outputs = self.text_decoder.generate(
                input_ids=bos_ids,
                max_length=50,
                min_length=1,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                encoder_hidden_states=question_states,
                encoder_attention_mask=question_atts,
            )

            answers = []
            for output in outputs:
                answer = self.tokenizer.decode(output, skip_special_tokens=True)
                answers.append(answer)
            return answers


def blip_vqa(pretrained='', **kwargs):
    from models.blip import load_checkpoint

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


def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key: str):
    """
    https://github.com/salesforce/BLIP/blob/main/models/blip_pretrain.py
    """
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        print(
            f"{decoder.__class__} and {encoder.__class__} are not equal. "
            f"In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
            decoder_pointer: nn.Module,
            encoder_pointer: nn.Module,
            module_name: str,
            uninitialized_encoder_weights: List[str],
            skip_key: str,
            depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            print(module_name + ' is tied')
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                    len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                            encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. "
                        "It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)
