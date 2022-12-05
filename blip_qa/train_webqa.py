import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.distributed as dist
from models.blip_webqa import blip_vqa
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.webqa_dataset import webqa_collate_fn
import wandb
import time
from torch.cuda import amp


def init_wandb(output_dir: str):
    # need to change to your own API when using
    os.environ['EXP_NUM'] = 'WebQA'
    os.environ['WANDB_NAME'] = time.strftime(
        '%Y-%m-%d %H:%M:%S',
        time.localtime(int(round(time.time() * 1000)) / 1000)
    )
    os.environ['WANDB_API_KEY'] = 'b6bb57b85f5b5386441e06a96b564c28e96d0733'
    os.environ['WANDB_DIR'] = output_dir
    wandb.init(project="blip_webqa_qa")


def train(config, args, model, train_loader, val_loader, optimizer, epoch_start: int, global_step: int, device):
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    grad_accum = config['grad_accum']
    assert grad_accum >= 1
    grad_clip = config['grad_clip']
    assert grad_clip > 0
    alpha = config['alpha']
    assert 0 <= alpha <= 1

    scaler = amp.GradScaler()

    print(f"Start training from epoch {epoch_start}")
    for epoch in range(epoch_start, config['max_epoch']):
        # """
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

        batch_size = config['batch_size_train']
        avg_loss = 0.0
        avg_qa_loss = 0.0
        avg_retr_loss = 0.0

        model.train()
        for i, (
                images, captions, question, answer, n_img_facts, _, _, retr_labels,
        ) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            retr_labels = torch.cat(retr_labels).to(device, non_blocking=True)

            with amp.autocast():
                qa_loss, retr, _ = model(images, captions, question, answer, n_img_facts, train=True)

                # Retrieval loss
                if args.multitask:
                    retr_preds = [retr[i, :nf] for i, nf in enumerate(n_img_facts)]
                    retr_preds = torch.cat(retr_preds)
                    retr_loss = F.binary_cross_entropy_with_logits(
                        retr_preds, retr_labels, reduction='sum'
                    ) / images.size(0)

                    # overall loss
                    loss = (1 - alpha) * qa_loss + alpha * retr_loss

                    avg_qa_loss += qa_loss.item() * batch_size
                    avg_retr_loss += retr_loss.item() * batch_size
                else:
                    loss = qa_loss

                avg_loss += loss.item() * batch_size

                # grad accum
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (i + 1) % grad_accum == 0:
                # gradient clip
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                # optimize
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                global_step += 1

                # log avg losses
                avg_qa_loss /= grad_accum * batch_size
                avg_retr_loss /= grad_accum * batch_size
                avg_loss /= grad_accum * batch_size
                print(f'Epoch[{epoch}] step {global_step}:'
                      f'\tloss {avg_loss:.4f}\tqa_loss {avg_qa_loss:.4f}\tretr_loss {avg_retr_loss:.4f}')
                wandb.log({
                    f'loss': avg_loss, 'qa_loss': avg_qa_loss, 'retr_loss': avg_retr_loss, 'step': global_step,
                })

                # reset
                avg_qa_loss = 0.0
                avg_retr_loss = 0.0
                avg_loss = 0.0

        if utils.is_main_process():
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'global_step': global_step,
            }
            torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{epoch:02d}.pth'))

        if args.distributed:
            dist.barrier()
        # """

        # evaluation
        if utils.is_main_process():
            metric = evaluation(model, val_loader, device)

            wandb.log({'eval_acc': metric['acc'], 'step': global_step})
            wandb.log({'eval_FL': metric['fl'], 'step': global_step})
            wandb.log({'eval_qa': metric['acc'] * metric['fl'], 'step': global_step})


@torch.no_grad()
def evaluation(model, data_loader, device):
    from tqdm import tqdm
    from calculate_qa_metrics import calc_qa_metrics

    print('Start evaluation')
    refs = []
    preds = []
    qcates = []
    model.eval()
    val_iter = tqdm(data_loader, desc="Validation", disable=0)
    for i, (
            images, captions, question, answer, n_img_facts, question_ids, qcate, _,
    ) in enumerate(val_iter):
        images = images.to(device, non_blocking=True)
        with amp.autocast():
            pred = model(images, captions, question, answer, n_img_facts, train=False)

        preds += pred
        refs += answer
        qcates += qcate

    return calc_qa_metrics(preds, refs, qcates)


@torch.no_grad()
def inference(config, model, data_loader, device):
    from tqdm import tqdm

    model.eval()

    print("Start inference")
    result = []
    data_iter = tqdm(data_loader, desc="Validation", disable=0)
    for i, (
            images, captions, question, answer, n_img_facts, question_ids, qcates, _,
    ) in enumerate(data_iter):
        images = images.to(device, non_blocking=True)
        with amp.autocast():
            pred = model(images, captions, question, answer, n_img_facts, train=False)

        for ans, p, qid, qcate in zip(answer, pred, question_ids, qcates):
            result.append({"question_id": qid, 'qcate': qcate, "pred": p, "answer": ans})

    return result


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #### Dataset #### 
    print("Creating WebQA datasets")
    datasets = create_dataset(config, max_n_neg_facts=4 if args.multitask else 0)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(
        datasets, samplers,
        batch_size=[config['batch_size_train'], config['batch_size_val'], config['batch_size_test']],
        num_workers=[4, 4, 4], is_trains=[True, False, False],
        collate_fns=[webqa_collate_fn, webqa_collate_fn, webqa_collate_fn]
    )

    #### Model and optimizer ####
    epoch = 0
    global_step = 0
    optimizer_state = None
    print("Creating model")
    if args.resume:
        obj = torch.load(args.resume, map_location='cpu')
        config = obj['config']
        epoch = obj['epoch'] + 1
        global_step = obj['global_step'] + 1

        model = blip_vqa(
            image_size=config['image_size'],
            vit=config['vit'],
            vit_grad_ckpt=config['vit_grad_ckpt'],
            vit_ckpt_layer=config['vit_ckpt_layer'],
            multitask=args.multitask,
        )
        model.load_state_dict(obj['model'])

        optimizer_state = obj['optimizer']
    else:
        model = blip_vqa(
            pretrained=config['pretrained'],
            image_size=config['image_size'],
            vit=config['vit'],
            vit_grad_ckpt=config['vit_grad_ckpt'],
            vit_ckpt_layer=config['vit_ckpt_layer'],
            multitask=args.multitask,
        )
    model = model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    if args.inference:
        # inference split
        inference_split = args.inference_split
        if inference_split == 'val':
            result = inference(config, model, val_loader, device)
        elif inference_split == 'test':
            result = inference(config, model, test_loader, device)
        else:
            raise RuntimeError(f"Invalid --inference_split: {inference_split}")

        result_file = os.path.join(args.output_dir, f'{args.inference_split}_pred.json')
        with open(result_file, 'w') as f:
            json.dump(result, f)
        print(f'result file saved to {result_file}')
    else:
        # init wandb
        wandb_output_dir = os.path.join(args.output_dir, 'wandb')
        os.makedirs(wandb_output_dir, exist_ok=True)
        init_wandb(wandb_output_dir)

        train(config, args, model, train_loader, val_loader, optimizer, epoch, global_step, device)


def load_args_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/webqa.yaml')
    parser.add_argument('--multitask', action='store_true', default=False)
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_split', type=str, default='val')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    return args, config


if __name__ == '__main__':
    main(*load_args_configs())
