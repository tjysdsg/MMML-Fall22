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


@torch.no_grad()
def viz_att(config, args, model, train_loader, device):
    model.eval()
    for i, (images, captions, question, answer, n_facts, _, _,) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        loss, outputs = model(images, captions, question, answer, n_facts, train=True)

        if outputs.cross_att is None:
            print("[ERROR] Please add '\"output_attentions\": true' to the end of configs/med_config.json")
            exit(1)

        print(outputs.cross_att.shape)
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
        batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
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

    viz_att(config, args, model, train_loader, device)


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
