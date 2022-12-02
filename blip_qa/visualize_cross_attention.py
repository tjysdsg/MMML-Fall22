import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import torch
from models.blip_webqa import blip_vqa
import utils
from data import create_dataset, create_loader
from data.webqa_dataset import webqa_collate_fn
from matplotlib import pyplot as plt


@torch.no_grad()
def viz_att(config, args, model, train_loader, device):
    model.eval()
    for i, (images, captions, question, answer, n_facts, _, _,) in enumerate(train_loader):
        if n_facts[0] <= 1:
            continue

        images = images.to(device, non_blocking=True)

        # atts = (batch, num_heads, question_len, img_embed_len)
        loss, atts = model(images, captions, question, answer, n_facts, output_attentions=True)

        atts = torch.mean(atts[0], dim=0)  # (question_len, img_embed_len)

        # num_patches = model.visual_encoder.patch_embed.num_patches + 1
        atts = atts.view(atts.shape[0], n_facts[0], -1)  # (question_len, n_facts, num_patches)

        att_mean = torch.mean(atts, dim=-1)  # (question_len, n_facts)
        att_std = torch.std(atts, dim=-1)  # (question_len, n_facts)

        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        ax = axes.flat
        im = ax[0].imshow(att_mean.detach().cpu().numpy())
        ax[0].set_title('Mean attention scores')
        im = ax[1].imshow(att_std.detach().cpu().numpy())
        ax[1].set_title('Std. attention scores')
        fig.colorbar(im, ax=axes.ravel().tolist())
        plt.savefig('shit.jpg')
        plt.close('all')

        exit(0)


def main(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("Creating WebQA datasets")
    datasets = create_dataset(config)
    train_loader, val_loader, test_loader = create_loader(
        datasets, [None, None, None],
        batch_size=[1, 1, 1],
        num_workers=[1, 1, 1], is_trains=[True, False, False],
        collate_fns=[webqa_collate_fn, webqa_collate_fn, webqa_collate_fn]
    )

    print("Loading model")
    if args.resume:
        obj = torch.load(args.resume, map_location='cpu')
        config = obj['config']

        model = blip_vqa(
            image_size=config['image_size'],
            vit=config['vit'],
            vit_grad_ckpt=config['vit_grad_ckpt'],
            vit_ckpt_layer=config['vit_ckpt_layer'],
        )
        model.load_state_dict(obj['model'])
    else:
        model = blip_vqa(
            pretrained=config['pretrained'],
            image_size=config['image_size'],
            vit=config['vit'],
            vit_grad_ckpt=config['vit_grad_ckpt'],
            vit_ckpt_layer=config['vit_ckpt_layer'],
        )
    model = model.to(device)

    viz_att(config, args, model, val_loader, device)


def load_args_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/webqa.yaml')
    parser.add_argument('--output_dir', default='output/WebQA')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    return args, config


if __name__ == '__main__':
    main(*load_args_configs())
