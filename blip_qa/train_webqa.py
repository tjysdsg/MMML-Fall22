import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import json
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from models.blip_webqa import blip_vqa
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.webqa_dataset import webqa_collate_fn


def train(config, args, model, train_loader, device):
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    print("Start training")
    for epoch in range(0, config['max_epoch']):
        if not args.inference:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

            train_stats = train_1epoch(config, model, train_loader, optimizer, epoch, device)
        else:
            break

        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))

        if args.distributed:
            dist.barrier()


def train_1epoch(config, model, data_loader, optimizer, epoch, device):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = f'Train Epoch: [{epoch}]'
    print_freq = 50

    grad_accum = config['grad_accum']
    assert grad_accum >= 1
    grad_clip = config['grad_clip']
    assert grad_clip > 0

    for i, (
            images, captions, question, answer, n_facts, _, _,
    ) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = images.to(device, non_blocking=True)
        loss = model(images, captions, question, answer, n_facts, train=True)

        loss = loss / grad_accum
        loss.backward()

        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: f"{meter.global_avg:.3f}" for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def inference(config, model, data_loader, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Inference:'
    print_freq = 50

    result = []
    for i, (images, captions, question, answer, n_facts, question_ids, qcates) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        images = images.to(device, non_blocking=True)
        pred = model(images, captions, question, answer, n_facts, train=False)

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
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating webqa datasets")
    datasets = create_dataset('webqa', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
    else:
        samplers = [None, None]

    train_loader, val_loader, test_loader = create_loader(
        datasets, samplers,
        batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
        num_workers=[4, 4, 4], is_trains=[True, False, False],
        collate_fns=[webqa_collate_fn, webqa_collate_fn, webqa_collate_fn]
    )

    #### Model ####
    print("Creating model")
    model = blip_vqa(pretrained=config['pretrained'], image_size=config['image_size'],
                     vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    model = model.to(device)

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
        train(config, args, model, train_loader, device)


def load_args_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/webqa.yaml')
    parser.add_argument('--output_dir', default='output/WebQA')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_split', type=str, default='val')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    return args, config


if __name__ == '__main__':
    main(*load_args_configs())
