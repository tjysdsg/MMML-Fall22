import argparse
import os
import ruamel_yaml as yaml
import numpy as np
from pathlib import Path
import torch
from data import create_dataset, create_loader
from data.webqa_dataset import webqa_collate_fn
from models.blip_webqa import blip_vqa
import torch.nn.functional as F
from matplotlib import pyplot as plt


@torch.no_grad()
def main(args, config):
    # dataset
    dataset, _, _ = create_dataset(
        dict(
            image_size=480,
            train_file=r'E:\webqa\data\WebQA_train_val.json',
            val_file=r'E:\webqa\data\WebQA_train_val.json',
            test_file=r'E:\webqa\data\WebQA_train_val.json',
            image_dir=r'E:\webqa\data\images',
        ),
        use_num_samples=100,
    )
    train_loader = create_loader(
        [dataset], [None],
        batch_size=[2],
        num_workers=[1], is_trains=[True],
        collate_fns=[webqa_collate_fn]
    )[0]

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = blip_vqa(
        pretrained=r'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth',
        image_size=config['image_size'],
        vit=config['vit'],
        vit_grad_ckpt=config['vit_grad_ckpt'],
        vit_ckpt_layer=config['vit_ckpt_layer'],
    )
    model = model.to(device)

    for images, captions, question, answer, n_img_facts, question_ids, qcates, retr_labels in train_loader:
        print('QUESTION:', question)
        print('CAPTIONS:', captions)
        print('ANSWER:', answer)

        # visualize images
        # batch_size, nf, channel, H, W = images.shape

        # # print(f'(batch_size, n_img_facts, channel, H, W):', images.shape)
        # for b in range(batch_size):
        #     for fi in range(nf):
        #         im = images[b, fi].detach().cpu().numpy()
        #         im = np.transpose(im, (1, 2, 0))

        #         plt.imshow(im)
        #         plt.savefig(
        #             os.path.join(
        #                 args.output_dir,
        #                 f'{question_ids[b]}_{fi}.jpg',
        #             )
        #         )
        #         plt.close('all')

        # run model
        images = images.to(device, non_blocking=True)
        (
            loss, retr, multimodal_cross_atts
        ) = model(images, captions, question, answer, n_img_facts, train=True)

        # MULTITASK
        # retr_labels = torch.cat(retr_labels).to(device, non_blocking=True)
        # retr_preds = [retr[i, :nf] for i, nf in enumerate(n_img_facts)]
        # retr_preds = torch.cat(retr_preds)
        # retr_loss = F.binary_cross_entropy_with_logits(
        #     retr_preds, retr_labels, reduction='sum'
        # ) / images.size(0)

        # print('retr predictions', F.sigmoid(retr_preds))
        # print('retr labels', retr_labels)
        # print('retr loss', retr_loss)

        # for ans, p, qid, qcate in zip(answer, pred, question_ids, qcates):
        #     print({"question_id": qid, 'qcate': qcate, "pred": p, "answer": ans})


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
