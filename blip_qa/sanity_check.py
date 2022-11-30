import argparse
import os
import ruamel_yaml as yaml
import numpy as np
from pathlib import Path
import torch
from data import create_dataset, create_loader
from data.webqa_dataset import webqa_collate_fn
from models.blip_webqa import blip_vqa
from matplotlib import pyplot as plt


@torch.no_grad()
def main(args, config):
    # dataset
    datasets = create_dataset(config)
    train_loader, val_loader, test_loader = create_loader(
        datasets, [None, None, None],
        batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
        num_workers=[1, 1, 1], is_trains=[True, False, False],
        collate_fns=[webqa_collate_fn, webqa_collate_fn, webqa_collate_fn]
    )

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = blip_vqa(
        pretrained=config['pretrained'],
        image_size=config['image_size'],
        vit=config['vit'],
        vit_grad_ckpt=config['vit_grad_ckpt'],
        vit_ckpt_layer=config['vit_ckpt_layer'],
    )
    model = model.to(device)

    for i, (
            images, captions, question, answer, n_facts, question_ids, qcates
    ) in enumerate(train_loader):
        # visualize images
        batch_size, nf, channel, H, W = images.shape
        print(f'(batch_size, n_facts, channel, H, W):', images.shape)
        for b in range(batch_size):
            for fi in range(nf):
                im = images[b, fi].detach().cpu().numpy()
                im = np.transpose(im, (1, 2, 0))

                plt.imshow(im)
                plt.savefig(
                    os.path.join(
                        args.output_dir,
                        f'{question_ids[b]}_{fi}.jpg',
                    )
                )
                plt.close('all')

        # run model
        images = images.to(device, non_blocking=True)
        pred = model(images, captions, question, answer, n_facts, train=False)

        for ans, p, qid, qcate in zip(answer, pred, question_ids, qcates):
            print({"question_id": qid, 'qcate': qcate, "pred": p, "answer": ans})


def load_args_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/webqa.yaml')
    parser.add_argument('--output_dir', default='output/WebQA')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return args, config


if __name__ == '__main__':
    main(*load_args_configs())
