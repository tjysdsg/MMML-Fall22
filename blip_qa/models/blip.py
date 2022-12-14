'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import warnings

warnings.filterwarnings("ignore")
from models.vit import VisionTransformer, interpolate_pos_embed
from transformers import BertTokenizer
import torch
import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file


def init_tokenizer(cased: bool):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased' if cased else 'bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit == 'base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing,
                                           ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                           )
    elif vit == 'large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24,
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing,
                                           ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                           )
    return visual_encoder, vision_width


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_blip_state_dict(model, state_dict):
    if 'visual_encoder.pos_embed' in state_dict:
        state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(
            state_dict['visual_encoder.pos_embed'],
            model.visual_encoder,
        )
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(
            state_dict['visual_encoder_m.pos_embed'],
            model.visual_encoder_m,
        )

    def init_empty(k):
        v = torch.zeros_like(model.state_dict()[k])
        if 'bias' in k:
            torch.nn.init.zeros_(v)
        else:
            torch.nn.init.xavier_uniform(v)
        state_dict[k] = v

    for key in model.state_dict():
        if key in state_dict:
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]
        else:
            init_empty(key)

    msg = model.load_state_dict(state_dict, strict=False)
    return model, msg


def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']
    model, msg = load_blip_state_dict(model, state_dict)

    print(f'Loaded checkpoint from {url_or_filename}')
    return model, msg
